#include "m_pd.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

// forward declaration so pd_new() can see it
static t_class *exciter8_tilde_class;
/*
    exciter8~  — Universal 8-knob stereo exciter (MIDI-friendly version)
    Author: Juicy
    License: MIT

    Inlets (all float 0..1):
      1 Attack     — attack time (0.3–120 ms)
      2 Decay      — decay time (8–1500 ms)
      3 Sustain    — sustain level (0–1)
      4 Release    — release time (8–1500 ms)
      5 Hardness   — nonlinear compression strength (soft ↔ hydraulic press)
      6 Brightness — spectral tilt: 0=LP, 0.5=neutral, 1=HP
      7 ClickAmt   — impulse strike loudness (0 muted, 1 full hard strike)
      8 DC Amt     — steady pressure fuel (0 none, 1 full)

    Messages:
      bang         — trigger ADSR + click
      freq <Hz>    — tonal driver freq (optional)
      seed <u32>   — RNG seed
      hp <Hz>      — DC blocker cutoff (default 10 Hz)
*/

#define TWO_PI 6.28318530717958647692f
#define CLAMP01(x) ((x)<0.f?0.f:((x)>1.f?1.f:(x)))
#define LERP(a,b,t) ((a)+((b)-(a))*(t))

typedef struct _exciter8_tilde {
    t_object x_obj;
    t_outlet *outL, *outR;

    // parameters
    t_float p_attack, p_decay, p_sustain, p_release;
    t_float p_hardness, p_brightness, p_click, p_dcamt;

    float s_attack, s_decay, s_sustain, s_release;
    float s_hardness, s_brightness, s_click, s_dcamt;

    float sr, inv_sr;

    // envelope ADSR
    int env_state; // 0=idle,1=attack,2=decay,3=sustain,4=release
    float env_level;
    float env_a_inc, env_d_inc, env_r_inc;

    // click impulse
    int click_on;
    float click_phase, click_inc;
    int click_len;

    // tonal driver
    float note_hz;
    float tone_phase;

    // noise
    uint32_t rng;

    // filter states
    float lp_state, hp_state;

    // DC blocker
    float hp_a, hp_xL, hp_yL, hp_xR, hp_yR;

} t_exciter8_tilde;

// RNG
static inline uint32_t xs32(uint32_t *s){uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; return *s=x;}
static inline float white01(uint32_t *s){return (int32_t)xs32(s)*(1.0f/2147483648.0f);}

// math helpers
static inline float tanh_fast(float x){const float x2=x*x; return x*(27.0f+x2)/(27.0f+9.0f*x2);}
static inline float fold_triangle(float x){float y=x+1.f; y-=2.f*floorf(y*0.5f); y=fabsf(y-1.f); return (y*2.f)-1.f;}
static inline float nonlin_morph(float x,float hard){
    float drive=1.f+hard*4.f; // more intense than before
    float y0=x;
    float y1=fmaxf(-1.f,fminf(1.f,x*drive));
    float y2=tanh_fast(x*drive*1.5f);
    float y3=fold_triangle(x*drive*2.f);
    if(hard<0.33f){float t=hard/0.33f;return LERP(y0,y1,t);}
    else if(hard<0.66f){float t=(hard-0.33f)/0.33f;return LERP(y1,y2,t);}
    else{float t=(hard-0.66f)/0.34f;return LERP(y2,y3,t);}
}

static inline int ms2samps(float ms,float sr){return (int)(ms*0.001f*sr);}

static inline void hp_set_coeffs(t_exciter8_tilde *x,float hz){float c=expf(-TWO_PI*hz/x->sr); x->hp_a=c;}

// one-pole HP
static inline float hp_proc(float in,float *x1,float *y1,float a){float y=a*((*y1)+in-(*x1));*x1=in;*y1=y;return y;}

// brightness tilt: strong LP/HP
static inline float tilt_filter(float in,float *lp,float *hp,float b,float sr){
    float cutoff = LERP(200.f,8000.f,b);
    float a_lp = expf(-2.f*3.1415926f*cutoff/sr);
    *lp=(1.f-a_lp)*in + a_lp*(*lp);
    float hpv=in-*lp;
    if(b<0.5f){float amt=(1.f-b*2.f);return (*lp)*amt;}
    else if(b>0.5f){float amt=(b-0.5f)*2.f;return hpv*amt;}
    else return in;
}

// perform
static t_int *exciter8_tilde_perform(t_int *w){
    t_exciter8_tilde *x=(t_exciter8_tilde*)(w[1]);
    int n=(int)(w[2]);
    t_sample *outL=(t_sample*)(w[3]);
    t_sample *outR=(t_sample*)(w[4]);

    for(int i=0;i<n;i++){
        // envelope
        switch(x->env_state){
            case 1: x->env_level+=x->env_a_inc; if(x->env_level>=1.f){x->env_level=1.f;x->env_state=2;} break;
            case 2: x->env_level-=x->env_d_inc; if(x->env_level<=x->s_sustain){x->env_level=x->s_sustain;x->env_state=3;} break;
            case 3: x->env_level=x->s_sustain; break;
            case 4: x->env_level-=x->env_r_inc; if(x->env_level<=0.f){x->env_level=0.f;x->env_state=0;} break;
        }

        // click impulse
        float impulse=0.f;
        if(x->click_on){
            float w=0.5f-0.5f*cosf(TWO_PI*x->click_phase);
            impulse=w*x->s_click;
            x->click_phase+=x->click_inc;
            if(x->click_phase>=1.f) x->click_on=0;
        }

        float dc=x->s_dcamt;                   // steady pressure
        float noise=white01(&x->rng);          // noise
        x->tone_phase+=TWO_PI*(x->note_hz/x->sr);
        if(x->tone_phase>TWO_PI)x->tone_phase-=TWO_PI;
        float tone=sinf(x->tone_phase);       // optional tonal driver

        float mix=(impulse+dc+noise*0.5f+tone*0.0f)*x->env_level;
        float shaped=nonlin_morph(mix,x->s_hardness);
        float col=tilt_filter(shaped,&x->lp_state,&x->hp_state,x->s_brightness,x->sr);

        float nL=col+0.05f*white01(&x->rng);
        float nR=col+0.05f*white01(&x->rng);

        float yL=hp_proc(nL,&x->hp_xL,&x->hp_yL,x->hp_a);
        float yR=hp_proc(nR,&x->hp_xR,&x->hp_yR,x->hp_a);

        outL[i]=tanh_fast(yL*1.2f);
        outR[i]=tanh_fast(yR*1.2f);
    }
    return(w+5);
}

// dsp
static void exciter8_tilde_dsp(t_exciter8_tilde *x,t_signal **sp){
    x->sr=(float)sp[0]->s_sr; if(x->sr<=0)x->sr=44100.f; x->inv_sr=1.f/x->sr;
    dsp_add(exciter8_tilde_perform,4,x,sp[0]->s_n,sp[0]->s_vec,sp[1]->s_vec);
}

// note-on trigger
static void exciter8_tilde_bang(t_exciter8_tilde *x){
    x->env_state=1; x->env_level=0.f;
    float atk_ms=LERP(0.3f,120.f,x->s_attack);
    float dec_ms=LERP(8.f,1500.f,x->s_decay);
    float rel_ms=LERP(8.f,1500.f,x->s_release);
    x->env_a_inc=1.f/fmaxf(1,ms2samps(atk_ms,x->sr));
    x->env_d_inc=(1.f-x->s_sustain)/fmaxf(1,ms2samps(dec_ms,x->sr));
    x->env_r_inc=x->s_sustain/fmaxf(1,ms2samps(rel_ms,x->sr));
    // click
    float click_ms=3.0f;
    x->click_len=ms2samps(click_ms,x->sr);
    x->click_inc=(x->click_len>0)?1.f/(float)x->click_len:1.f;
    x->click_phase=0.f;
    x->click_on=(x->s_click>0.001f);
}

// methods
static void exciter8_tilde_freq(t_exciter8_tilde *x,t_floatarg f){if(f>0)x->note_hz=f;}
static void exciter8_tilde_seed(t_exciter8_tilde *x,t_floatarg s){uint32_t v=(uint32_t)(s<=0?1:s);x->rng=v;}
static void exciter8_tilde_hp(t_exciter8_tilde *x,t_floatarg hz){if(hz<1)hz=1;if(hz>60)hz=60;hp_set_coeffs(x,hz);}

// new/free
static void *exciter8_tilde_new(void){
    t_exciter8_tilde *x=(t_exciter8_tilde*)pd_new(exciter8_tilde_class);
    x->sr=sys_getsr(); if(x->sr<=0)x->sr=44100.f; x->inv_sr=1.f/x->sr;
    x->p_attack=0.1f;x->p_decay=0.2f;x->p_sustain=0.7f;x->p_release=0.3f;
    x->p_hardness=0.3f;x->p_brightness=0.5f;x->p_click=0.5f;x->p_dcamt=0.5f;
    x->s_attack=x->p_attack;x->s_decay=x->p_decay;x->s_sustain=x->p_sustain;
    x->s_release=x->p_release;x->s_hardness=x->p_hardness;x->s_brightness=x->p_brightness;
    x->s_click=x->p_click;x->s_dcamt=x->p_dcamt;
    x->note_hz=220.f;x->tone_phase=0.f;x->rng=1234567u;
    x->env_state=0;x->env_level=0.f;
    x->lp_state=0.f;x->hp_state=0.f; x->hp_xL=x->hp_yL=x->hp_xR=x->hp_yR=0.f;
    hp_set_coeffs(x,10.f);
    floatinlet_new(&x->x_obj,&x->p_attack);
    floatinlet_new(&x->x_obj,&x->p_decay);
    floatinlet_new(&x->x_obj,&x->p_sustain);
    floatinlet_new(&x->x_obj,&x->p_release);
    floatinlet_new(&x->x_obj,&x->p_hardness);
    floatinlet_new(&x->x_obj,&x->p_brightness);
    floatinlet_new(&x->x_obj,&x->p_click);
    floatinlet_new(&x->x_obj,&x->p_dcamt);
    x->outL=outlet_new(&x->x_obj,&s_signal);
    x->outR=outlet_new(&x->x_obj,&s_signal);
    return(void*)x;
}
static void exciter8_tilde_free(t_exciter8_tilde *x){}

// class
static t_class *exciter8_tilde_class;
void exciter8_tilde_setup(void){
    exciter8_tilde_class=class_new(gensym("exciter8~"),
        (t_newmethod)exciter8_tilde_new,
        (t_method)exciter8_tilde_free,
        sizeof(t_exciter8_tilde),CLASS_DEFAULT,0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_dsp,gensym("dsp"),A_CANT,0);
    class_addbang(exciter8_tilde_class,(t_method)exciter8_tilde_bang);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_freq,gensym("freq"),A_DEFFLOAT,0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_seed,gensym("seed"),A_DEFFLOAT,0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_hp,gensym("hp"),A_DEFFLOAT,0);
}
