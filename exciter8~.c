#include "m_pd.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/*  exciter8~ — velocity-aware stereo exciter (final)
    Inlets (1–9, left to right):
      1  [message inlet]: noteon <vel>, noteoff, mode impulse|dc, freq <Hz>, seed <u32>, hp <Hz>, bang
      2  Attack (ms)
      3  Decay (ms)
      4  Sustain (0..1)
      5  Release (ms)
      6  Hardness (0..1)
      7  Brightness (0..1)  // 0=LP, 0.5=neutral, 1=HP
      8  Click Amount (0..1)
      9  EnvToFilter Depth (0..1)

    Outlets: L, R audio
    License: MIT
*/

#define TWO_PI 6.28318530717958647692f
#define LERP(a,b,t) ((a)+((b)-(a))*(t))
#define CLAMP(x,lo,hi) ((x)<(lo)?(lo):((x)>(hi)?(hi):(x)))

static t_class *exciter8_tilde_class;

typedef struct _exciter8_tilde {
    t_object x_obj;
    t_outlet *outL, *outR;

    /* float params */
    t_float p_attack_ms, p_decay_ms, p_sustain, p_release_ms;
    t_float p_hardness, p_brightness, p_click, p_envfilt;

    float sr;

    /* ADSR */
    int   env_state;      /* 0 idle, 1 atk, 2 dec, 3 sus, 4 rel */
    float env_level;      /* 0..env_peak */
    float env_peak;       /* velocity */
    float env_a_inc, env_d_inc, env_r_inc;
    float note_vel;       /* 0..1 */

    /* strike burst */
    int   click_on;
    float click_phase, click_inc;
    int   exc_mode;       /* 0 impulse, 1 dc */

    /* optional oscillator (kept for future) */
    float note_hz;
    float tone_phase;

    /* RNG */
    uint32_t rng;

    /* pre-filters */
    float lp200_state;
    float hp8k_x1, hp8k_y1;

    /* output DC blockers */
    float dc_a, dc_xL, dc_yL, dc_xR, dc_yR;

} t_exciter8_tilde;

/* ----- utils ----- */
static inline uint32_t xs32(uint32_t *s){uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; return *s=x;}
static inline float white01(uint32_t *s){return (int32_t)xs32(s)*(1.0f/2147483648.0f);}

static inline float tanh_fast(float x){const float x2=x*x; return x*(27.0f+x2)/(27.0f+9.0f*x2);}
static inline float fold_triangle(float x){float y=x+1.f; y-=2.f*floorf(y*0.5f); y=fabsf(y-1.f); return (y*2.f)-1.f;}

static inline float nonlin_morph(float x, float hard){
    /* linear -> clip -> tanh -> fold */
    float drive = 1.f + hard*4.f;
    float y0 = x;
    float y1 = fmaxf(-1.f, fminf(1.f, x*drive));
    float y2 = tanh_fast(x*drive*1.5f);
    float y3 = fold_triangle(x*drive*2.f);
    if (hard < 0.33f){ float t=hard/0.33f; return LERP(y0,y1,t); }
    else if (hard < 0.66f){ float t=(hard-0.33f)/0.33f; return LERP(y1,y2,t); }
    else { float t=(hard-0.66f)/0.34f; return LERP(y2,y3,t); }
}

/* DC blocker */
static inline void dc_set_coeff(t_exciter8_tilde *x, float hz){ x->dc_a = expf(-TWO_PI * hz / x->sr); }
static inline float dc_hp(float in, float *x1, float *y1, float a){ float y = a*((*y1)+in-(*x1)); *x1=in; *y1=y; return y; }

/* 1-pole LP @ 200 Hz */
static inline float lp200(float in, float *state, float sr){
    const float cutoff = 200.f;
    const float a = expf(-2.f*3.1415926f*cutoff/sr);
    *state = (1.f - a)*in + a*(*state);
    return *state;
}
/* 1-pole HP @ 8 kHz */
static inline float hp8k(float in, float *x1, float *y1, float sr){
    const float cutoff = 8000.f;
    const float a = expf(-2.f*3.1415926f*cutoff/sr);
    float y = a*((*y1)+in-(*x1)); *x1=in; *y1=y; return y;
}

/* brightness + env depth -> weights for LP/Dry/HP (immediate blend) */
static inline void tilt_weights(float base_b, float env, float depth, float *w_lp, float *w_dry, float *w_hp){
    float b_mod = LERP(base_b, env, CLAMP(depth,0.f,1.f)); /* env=1 bright, env=0 dark if depth=1 */
    b_mod = CLAMP(b_mod, 0.f, 1.f);

    if (b_mod < 0.5f){
        float t = b_mod * 2.f; *w_lp = 1.f - t; *w_dry = t; *w_hp = 0.f;
    } else if (b_mod > 0.5f){
        float t = (b_mod - 0.5f) * 2.f; *w_lp = 0.f; *w_dry = 1.f - t; *w_hp = t;
    } else {
        *w_lp = 0.f; *w_dry = 1.f; *w_hp = 0.f;
    }
}

/* ----- perform ----- */
static t_int *exciter8_tilde_perform(t_int *w){
    t_exciter8_tilde *x = (t_exciter8_tilde *)(w[1]);
    int n = (int)(w[2]);
    t_sample *outL = (t_sample *)(w[3]);
    t_sample *outR = (t_sample *)(w[4]);

    for (int i = 0; i < n; i++){
        /* ADSR */
        switch (x->env_state){
            case 1:
                x->env_level += x->env_a_inc;
                if (x->env_level >= x->env_peak){ x->env_level = x->env_peak; x->env_state = 2; }
                break;
            case 2:
                x->env_level -= x->env_d_inc;
                if (x->env_level <= x->p_sustain * x->env_peak){
                    x->env_level = x->p_sustain * x->env_peak; x->env_state = 3;
                }
                break;
            case 3: /* sustain */ break;
            case 4:
                x->env_level -= x->env_r_inc;
                if (x->env_level <= 0.f){ x->env_level = 0.f; x->env_state = 0; }
                break;
        }

        /* hard mute if env is zero */
        if (x->env_level <= 0.f && x->env_state == 0){
            x->lp200_state = 0.f;
            x->hp8k_x1 = 0.f; x->hp8k_y1 = 0.f;
            x->click_on = 0;
            outL[i] = 0.f; outR[i] = 0.f;
            continue;
        }

        /* strike burst */
        float burst = 0.f;
        if (x->click_on){
            if (x->exc_mode == 0){
                float w = 0.5f - 0.5f * cosf(TWO_PI * x->click_phase); /* Hann */
                burst = w * x->p_click * x->note_vel;
            } else {
                burst = (1.f - x->click_phase) * x->p_click * x->note_vel; /* DC burst */
            }
            x->click_phase += x->click_inc;
            if (x->click_phase >= 1.f) x->click_on = 0;
        }

        /* noise */
        float noise = white01(&x->rng);

        /* pre-shape */
        float dry = (burst + 0.5f * noise) * x->env_level;

        /* hardness */
        float shaped = nonlin_morph(dry, x->p_hardness);

        /* branches */
        float lp_branch  = lp200(shaped, &x->lp200_state, x->sr);
        float dry_branch = shaped;
        float hp_branch  = hp8k  (shaped, &x->hp8k_x1, &x->hp8k_y1, x->sr);

        /* weights */
        float w_lp, w_dry, w_hp;
        tilt_weights(x->p_brightness, x->env_level, x->p_envfilt, &w_lp, &w_dry, &w_hp);

        float col = w_lp*lp_branch + w_dry*dry_branch + w_hp*hp_branch;

        /* tiny stereo decorrelation */
        float addL = 0.02f * white01(&x->rng);
        float addR = 0.02f * white01(&x->rng);

        /* DC block + safety */
        float yL = dc_hp(col + addL, &x->dc_xL, &x->dc_yL, x->dc_a);
        float yR = dc_hp(col + addR, &x->dc_xR, &x->dc_yR, x->dc_a);

        outL[i] = tanh_fast(yL * 1.2f);
        outR[i] = tanh_fast(yR * 1.2f);
    }
    return (t_int *)(w + 5);
}

/* ----- dsp add ----- */
static void exciter8_tilde_dsp(t_exciter8_tilde *x, t_signal **sp){
    x->sr = (float)sp[0]->s_sr; if (x->sr <= 0) x->sr = 44100.f;
    dsp_add(exciter8_tilde_perform, 4, x, sp[0]->s_n, sp[0]->s_vec, sp[1]->s_vec);
}

/* ----- note events ----- */
static void exciter8_tilde_noteon(t_exciter8_tilde *x, t_floatarg vel){
    float v = CLAMP(vel, 0.f, 1.f);
    if (v <= 0.f) return;

    x->note_vel  = v;
    x->env_peak  = v;
    x->env_state = 1;
    x->env_level = 0.f;

    /* velocity speeds attack; decay/release as set */
    float atk_samps = fmaxf(1.f, (x->p_attack_ms  * 0.001f * x->sr) / (0.25f + 0.75f * v));
    float dec_samps = fmaxf(1.f,  x->p_decay_ms   * 0.001f * x->sr);
    float rel_samps = fmaxf(1.f,  x->p_release_ms * 0.001f * x->sr);

    x->env_a_inc = x->env_peak / atk_samps;
    x->env_d_inc = (x->env_peak - x->p_sustain * x->env_peak) / dec_samps;
    x->env_r_inc = (x->p_sustain * x->env_peak) / rel_samps;

    /* strike window ~3 ms */
    x->click_inc   = 1.f / fmaxf(1.f, 3e-3f * x->sr);
    x->click_phase = 0.f;
    x->click_on    = (x->p_click > 0.001f);
}
static void exciter8_tilde_noteoff(t_exciter8_tilde *x){
    if (x->env_state == 0) return;
    x->env_state = 4;
}

/* ----- other methods ----- */
static void exciter8_tilde_mode(t_exciter8_tilde *x, t_symbol *s){
    if (!s) return;
    if (!strcmp(s->s_name, "impulse")) x->exc_mode = 0;
    else if (!strcmp(s->s_name, "dc")) x->exc_mode = 1;
}
static void exciter8_tilde_freq(t_exciter8_tilde *x, t_floatarg f){ if (f>0) x->note_hz = f; }
static void exciter8_tilde_seed(t_exciter8_tilde *x, t_floatarg s){ uint32_t v=(uint32_t)(s<=0?1:s); x->rng=v; }
static void exciter8_tilde_hp  (t_exciter8_tilde *x, t_floatarg hz){ dc_set_coeff(x, CLAMP(hz, 1.f, 60.f)); }
static void exciter8_tilde_bang(t_exciter8_tilde *x){ exciter8_tilde_noteon(x, 1.f); }

/* ----- new/free ----- */
static void *exciter8_tilde_new(void){
    t_exciter8_tilde *x = (t_exciter8_tilde *)pd_new(exciter8_tilde_class);

    x->sr = sys_getsr(); if (x->sr <= 0) x->sr = 44100.f;

    /* defaults */
    x->p_attack_ms  = 10.f;
    x->p_decay_ms   = 100.f;
    x->p_sustain    = 0.7f;
    x->p_release_ms = 200.f;
    x->p_hardness   = 0.35f;
    x->p_brightness = 0.5f;  /* neutral */
    x->p_click      = 0.6f;
    x->p_envfilt    = 0.5f;  /* env moves brightness toward LP on release */

    x->env_state = 0; x->env_level = 0.f; x->env_peak = 1.f; x->note_vel = 1.f;
    x->click_on = 0; x->click_phase = 0.f; x->click_inc = 1.f;
    x->exc_mode = 0; /* impulse */

    x->note_hz = 220.f; x->tone_phase = 0.f;
    x->rng = 1234567u;

    x->lp200_state = 0.f; x->hp8k_x1 = 0.f; x->hp8k_y1 = 0.f;

    x->dc_xL = x->dc_yL = x->dc_xR = x->dc_yR = 0.f;
    dc_set_coeff(x, 10.f);

    /* 8 float inlets */
    floatinlet_new(&x->x_obj, &x->p_attack_ms);
    floatinlet_new(&x->x_obj, &x->p_decay_ms);
    floatinlet_new(&x->x_obj, &x->p_sustain);
    floatinlet_new(&x->x_obj, &x->p_release_ms);
    floatinlet_new(&x->x_obj, &x->p_hardness);
    floatinlet_new(&x->x_obj, &x->p_brightness);
    floatinlet_new(&x->x_obj, &x->p_click);
    floatinlet_new(&x->x_obj, &x->p_envfilt);

    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);
    return (void *)x;
}
static void exciter8_tilde_free(t_exciter8_tilde *x){ (void)x; }

/* ----- setup ----- */
void exciter8_tilde_setup(void){
    exciter8_tilde_class = class_new(gensym("exciter8~"),
        (t_newmethod)exciter8_tilde_new,
        (t_method)exciter8_tilde_free,
        sizeof(t_exciter8_tilde), CLASS_DEFAULT, 0);

    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_dsp,      gensym("dsp"),     A_CANT, 0);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_noteon,   gensym("noteon"), A_DEFFLOAT, 0);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_noteoff,  gensym("noteoff"), 0);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_mode,     gensym("mode"),   A_SYMBOL, 0);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_freq,     gensym("freq"),   A_DEFFLOAT, 0);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_seed,     gensym("seed"),   A_DEFFLOAT, 0);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_hp,       gensym("hp"),     A_DEFFLOAT, 0);
    class_addbang   (exciter8_tilde_class, (t_method)exciter8_tilde_bang);
}
