#include "m_pd.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/*
    exciter8~  — Velocity-aware stereo exciter
    Author: Juicy
    License: MIT

    Inlets:
      1 Attack (ms)
      2 Decay (ms)
      3 Sustain (0..1)
      4 Release (ms)
      5 Hardness (0..1)
      6 Brightness (0..1)
      7 Click Amount (0..1)
      8 Env→Filter Depth (0..1)

    Messages:
      noteon <vel>  — start ADSR with velocity (0..1)
      noteoff       — release ADSR
      mode impulse  — impulse strike
      mode dc       — DC burst strike
      freq <Hz>     — tonal driver freq
      seed <u32>    — RNG seed
      hp <Hz>       — DC blocker cutoff
*/

#define TWO_PI 6.28318530717958647692f
#define LERP(a,b,t) ((a)+((b)-(a))*(t))
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

static t_class *exciter8_tilde_class;

typedef struct _exciter8_tilde {
    t_object x_obj;
    t_outlet *outL, *outR;

    // parameters
    t_float p_attack, p_decay, p_sustain, p_release;
    t_float p_hardness, p_brightness, p_click, p_envfilt;

    float sr;

    // ADSR state
    int env_state; // 0 idle, 1 atk, 2 dec, 3 sus, 4 rel
    float env_level;
    float env_a_inc, env_d_inc, env_r_inc;
    float env_peak;
    float note_vel;

    // click/DC burst
    int click_on;
    float click_phase, click_inc;
    int exc_mode; // 0 = impulse, 1 = DC

    // tonal driver
    float note_hz;
    float tone_phase;

    // RNG
    uint32_t rng;

    // filters
    float lp_state, hp_state;
    float hp_a, hp_xL, hp_yL, hp_xR, hp_yR;

} t_exciter8_tilde;

// RNG
static inline uint32_t xs32(uint32_t *s){uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; return *s=x;}
static inline float white01(uint32_t *s){return (int32_t)xs32(s)*(1.0f/2147483648.0f);}

// math
static inline float tanh_fast(float x){const float x2=x*x; return x*(27.0f+x2)/(27.0f+9.0f*x2);}
static inline float fold_triangle(float x){float y=x+1.f; y-=2.f*floorf(y*0.5f); y=fabsf(y-1.f); return (y*2.f)-1.f;}
static inline float nonlin_morph(float x,float hard){
    float drive=1.f+hard*4.f;
    float y0=x;
    float y1=fmaxf(-1.f,fminf(1.f,x*drive));
    float y2=tanh_fast(x*drive*1.5f);
    float y3=fold_triangle(x*drive*2.f);
    if(hard<0.33f){float t=hard/0.33f;return LERP(y0,y1,t);}
    else if(hard<0.66f){float t=(hard-0.33f)/0.33f;return LERP(y1,y2,t);}
    else{float t=(hard-0.66f)/0.34f;return LERP(y2,y3,t);}
}

static inline void hp_set_coeffs(t_exciter8_tilde *x,float hz){float c=expf(-TWO_PI*hz/x->sr); x->hp_a=c;}
static inline float hp_proc(float in,float *x1,float *y1,float a){float y=a*((*y1)+in-(*x1));*x1=in;*y1=y;return y;}

// tilt filter with env depth
static inline float tilt_filter_env(float in,float *lp,float *hp,float b,float env,float depth,float sr){
    float cutoff = LERP(200.f,8000.f,b);
    float a_lp = expf(-2.f*3.1415926f*cutoff/sr);
    *lp=(1.f-a_lp)*in + a_lp*(*lp);
    float hpv=in-*lp;

    float env_b = b;
    if(depth>0.f){
        env_b = LERP(1.f,0.f,depth*(1.f-env)); // bright→dark as env decays
    }

    if(env_b<0.5f){float amt=(1.f-env_b*2.f);return (*lp)*amt;}
    else if(env_b>0.5f){float amt=(env_b-0.5f)*2.f;return hpv*amt;}
    else return in;
}

// perform
static t_int *exciter8_tilde_perform(t_int *w){
    t_exciter8_tilde *x=(t_exciter8_tilde*)(w[1]);
    int n=(int)(w[2]);
    t_sample *outL=(t_sample*)(w[3]);
    t_sample *outR=(t_sample*)(w[4]);

    for(int i=0;i<n;i++){
        // ADSR
        switch(x->env_state){
            case 1: x->env_level+=x->env_a_inc; if(x->env_level>=x->env_peak){x->env_level=x->env_peak;x->env_state=2;} break;
            case 2: x->env_level-=x->env_d_inc; if(x->env_level<=x->p_sustain*x->env_peak){x->env_level=x->p_sustain*x->env_peak;x->env_state=3;} break;
            case 3: break; // sustain
            case 4: x->env_level-=x->env_r_inc; if(x->env_level<=0.f){x->env_level=0.f;x->env_state=0;} break;
        }

        float exc=0.f;
        if(x->click_on){
            if(x->exc_mode==0){ // impulse
                float w=0.5f-0.5f*cosf(TWO_PI*x->click_phase);
                exc=w*x->p_click*x->note_vel;
            }else{ // dc burst
                exc=x->p_click*(1.f-x->click_phase)*x->note_vel;
            }
            x->click_phase+=x->click_inc;
            if(x->click_phase>=1.f) x->click_on=0;
        }

        float noise=white01(&x->rng);
        x->tone_phase+=TWO_PI*(x->note_hz/x->sr);
        if(x->tone_phase>TWO_PI)x->tone_phase-=TWO_PI;
        float tone=sinf(x->tone_phase);

        float mix=(exc+noise*0.5f+tone*0.0f)*x->env_level;
        float shaped=nonlin_morph(mix,x->p_hardness);
        float col=tilt_filter_env(shaped,&x->lp_state,&x->hp_state,
                                  x->p_brightness,x->env_level,x->p_envfilt,x->sr);

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
    x->sr=(float)sp[0]->s_sr; if(x->sr<=0)x->sr=44100.f;
    dsp_add(exciter8_tilde_perform,4,x,sp[0]->s_n,sp[0]->s_vec,sp[1]->s_vec);
}

// noteon/off
static void exciter8_tilde_noteon(t_exciter8_tilde *x, t_floatarg vel){
    if(vel<=0) return;
    x->note_vel=CLAMP(vel,0.f,1.f);
    x->env_state=1; x->env_level=0.f;
    x->env_peak=x->note_vel;
    float atk_samps=fmaxf(1,(x->p_attack*0.001f*x->sr)/(0.2f+vel)); // faster if vel high
    float dec_samps=fmaxf(1,(x->p_decay*0.001f*x->sr));
    float rel_samps=fmaxf(1,(x->p_release*0.001f*x->sr));
    x->env_a_inc=x->env_peak/atk_samps;
    x->env_d_inc=(x->env_peak-x->p_sustain*x->env_peak)/dec_samps;
    x->env_r_inc=(x->p_sustain*x->env_peak)/rel_samps;
    x->click_inc=1.f/(3e-3f*x->sr);
    x->click_phase=0.f;
    x->click_on=(x->p_click>0.001f);
}
static void exciter8_tilde_noteoff(t_exciter8_tilde *x){x->env_state=4;}

// methods
static void exciter8_tilde_freq(t_exciter8_tilde *x,t_floatarg f){if(f>0)x->note_hz=f;}
static void exciter8_tilde_seed(t_exciter8_tilde *x,t_floatarg s){uint32_t v=(uint32_t)(s<=0?1:s);x->rng=v;}
static void exciter8_tilde_hp(t_exciter8_tilde *x,t_floatarg hz){if(hz<1)hz=1;if(hz>60)hz=60;hp_set_coeffs(x,hz);}
static void exciter8_tilde_mode(t_exciter8_tilde *x, t_symbol *s){
    if(strcmp(s->s_name,"impulse")==0) x->exc_mode=0;
    else if(strcmp(s->s_name,"dc")==0) x->exc_mode=1;
}

// new/free
static void *exciter8_tilde_new(void){
    t_exciter8_tilde *x=(t_exciter8_tilde*)pd_new(exciter8_tilde_class);
    x->sr=sys_getsr(); if(x->sr<=0)x->sr=44100.f;

    x->p_attack=10.f; x->p_decay=100.f; x->p_sustain=0.7f; x->p_release=200.f;
    x->p_hardness=0.3f; x->p_brightness=0.5f; x->p_click=0.5f; x->p_envfilt=0.0f;

    x->note_hz=220.f; x->tone_phase=0.f; x->rng=1234567u;
    x->env_state=0; x->env_level=0.f; x->note_vel=1.f;
    x->lp_state=0.f; x->hp_state=0.f; x->hp_xL=x->hp_yL=x->hp_xR=x->hp_yR=0.f;
    hp_set_coeffs(x,10.f);
    x->exc_mode=0;

    floatinlet_new(&x->x_obj,&x->p_attack);
    floatinlet_new(&x->x_obj,&x->p_decay);
    floatinlet_new(&x->x_obj,&x->p_sustain);
    floatinlet_new(&x->x_obj,&x->p_release);
    floatinlet_new(&x->x_obj,&x->p_hardness);
    floatinlet_new(&x->x_obj,&x->p_brightness);
    floatinlet_new(&x->x_obj,&x->p_click);
    floatinlet_new(&x->x_obj,&x->p_envfilt);

    x->outL=outlet_new(&x->x_obj,&s_signal);
    x->outR=outlet_new(&x->x_obj,&s_signal);
    return(void*)x;
}
static void exciter8_tilde_free(t_exciter8_tilde *x){}

// class
void exciter8_tilde_setup(void){
    exciter8_tilde_class=class_new(gensym("exciter8~"),
        (t_newmethod)exciter8_tilde_new,
        (t_method)exciter8_tilde_free,
        sizeof(t_exciter8_tilde),CLASS_DEFAULT,0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_dsp,gensym("dsp"),A_CANT,0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_noteon,gensym("noteon"),A_DEFFLOAT,0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_noteoff,gensym("noteoff"),0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_freq,gensym("freq"),A_DEFFLOAT,0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_seed,gensym("seed"),A_DEFFLOAT,0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_hp,gensym("hp"),A_DEFFLOAT,0);
    class_addmethod(exciter8_tilde_class,(t_method)exciter8_tilde_mode,gensym("mode"),A_SYMBOL,0);
}
