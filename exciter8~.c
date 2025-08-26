#include "m_pd.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/*  exciter8~ — Velocity-aware stereo exciter (final)
    Inlets (1–9):
      1  [message inlet]: noteon <vel>, noteoff, mode impulse|dc, freq <Hz>, seed <u32>, hp <Hz>, bang
      2  Attack (ms)
      3  Decay (ms)
      4  Sustain (0..1)
      5  Release (ms)
      6  Hardness (0..1)
      7  Brightness (0..1)  // 0=LP, 0.5=neutral, 1=HP
      8  Click Amount (0..1)
      9  Env→Filter Depth (0..1)

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

    // parameters (float inlets)
    t_float p_attack_ms, p_decay_ms, p_sustain, p_release_ms;
    t_float p_hardness, p_brightness, p_click, p_envfilt;

    // sample rate
    float sr;

    // ADSR state
    int   env_state;         // 0 idle, 1 atk, 2 dec, 3 sus, 4 rel
    float env_level;         // current env value (0..env_peak)
    float env_peak;          // velocity-scaled peak
    float env_a_inc, env_d_inc, env_r_inc;
    float note_vel;          // 0..1

    // strike burst (impulse/DC)
    int   click_on;
    float click_phase, click_inc;
    int   exc_mode;          // 0=impulse, 1=dc

    // tonal driver (optional; kept for future)
    float note_hz;
    float tone_phase;

    // RNG
    uint32_t rng;

    // pre-filter states (LP 200 Hz & HP 8 kHz)
    float lp200_state;
    float hp8k_x1, hp8k_y1;

    // output DC blocker per channel
    float dc_a, dc_xL, dc_yL, dc_xR, dc_yR;

} t_exciter8_tilde;

/* ===== utils ===== */
static inline uint32_t xs32(uint32_t *s){uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; return *s=x;}
static inline float white01(uint32_t *s){return (int32_t)xs32(s)*(1.0f/2147483648.0f);}

static inline float tanh_fast(float x){const float x2=x*x; return x*(27.0f+x2)/(27.0f+9.0f*x2);}
static inline float fold_triangle(float x){float y=x+1.f; y-=2.f*floorf(y*0.5f); y=fabsf(y-1.f); return (y*2.f)-1.f;}

static inline float nonlin_morph(float x, float hard){
    // linear → clip → tanh → fold (potent at 1.0)
    float drive = 1.f + hard*4.f;
    float y0 = x;
    float y1 = fmaxf(-1.f, fminf(1.f, x*drive));      // clip
    float y2 = tanh_fast(x*drive*1.5f);               // tanh
    float y3 = fold_triangle(x*drive*2.f);            // fold
    if (hard < 0.33f){ float t = hard/0.33f; return LERP(y0,y1,t); }
    else if (hard < 0.66f){ float t=(hard-0.33f)/0.33f; return LERP(y1,y2,t); }
    else { float t=(hard-0.66f)/0.34f; return LERP(y2,y3,t); }
}

/* DC blocker (per-channel one-pole HP) */
static inline void dc_set_coeff(t_exciter8_tilde *x, float hz){ x->dc_a = expf(-TWO_PI * hz / x->sr); }
static inline float dc_hp(float in, float *x1, float *y1, float a){ float y = a*((*y1)+in-(*x1)); *x1=in; *y1=y; return y; }

/* one-pole LP @ 200 Hz (strong low) */
static inline float lp200(float in, float *state, float sr){
    const float cutoff = 200.f;
    const float a = expf(-2.f*3.1415926f*cutoff/sr);
    *state = (1.f - a)*in + a*(*state);
    return *state;
}
/* one-pole HP @ 8 kHz (strong high) */
static inline float hp8k(float in, float *x1, float *y1, float sr){
    const float cutoff = 8000.f;
    const float a = expf(-2.f*3.1415926f*cutoff/sr);
    float y = a*((*y1)+in-(*x1)); *x1=in; *y1=y; return y;
}

/* Map Brightness + EnvDepth + Env to mix weights for LP/Dry/HP (immediate, no lag) */
static inline void tilt_weights(float base_b, float env, float depth, float *w_lp, float *w_dry, float *w_hp){
    // env_target goes HP at env=1, LP at env=0
    float env_target = env;               // 0..1 (0=LP side, 1=HP side)
    float b_mod = LERP(base_b, env_target, CLAMP(depth,0.f,1.f));
    b_mod = CLAMP(b_mod, 0.f, 1.f);

    // triangle mapping to LP/Dry/HP
    if (b_mod < 0.5f){
        float t = b_mod * 2.f;            // 0..1
        *w_lp = 1.f - t; *w_dry = t; *w_hp = 0.f;
    } else if (b_mod > 0.5f){
        float t = (b_mod - 0.5f) * 2.f;   // 0..1
        *w_lp = 0.f; *w_dry = 1.f - t; *w_hp = t;
    } else {
        *w_lp = 0.f; *w_dry = 1.f; *w_hp = 0.f;
    }
}

/* ===== DSP perform ===== */
static t_int *exciter8_tilde_perform(t_int *w){
    t_exciter8_tilde *x = (t_exciter8_tilde *)(w[1]);
    int n = (int)(w[2]);
    t_sample *outL = (t_sample *)(w[3]);
    t_sample *outR = (t_sample *)(w[4]);

    for (int i = 0; i < n; i++){
        /* ADSR step */
        switch (x->env_state){
            case 1: x->env_level += x->env_a_inc;
                    if (x->env_level >= x->env_peak){ x->env_level = x->env_peak; x->env_state = 2; }
                    break;
            case 2: x->env_level -= x->env_d_inc;
                    if (x->env_level <= x->p_sustain * x->env_peak){
                        x->env_level = x->p_sustain * x->env_peak; x->env_state = 3;
                    }
                    break;
            case 3: /* sustain */ break;
            case 4: x->env_level -= x->env_r_inc;
                    if (x->env_level <= 0.f){ x->env_level = 0.f; x->env_state = 0; }
                    break;
        }

        /* HARD MUTE when envelope is 0: no leak + reset states */
        if (x->env_level <= 0.f && x->env_state == 0){
            x->lp200_state = 0.f;
            x->hp8k_x1 = 0.f; x->hp8k_y1 = 0.f;
            x->click_on = 0;
            outL[i] = 0.f; outR[i] = 0.f;
            continue;
        }

        /* strike burst (impulse/DC) while click_on */
        float burst = 0.f;
        if (x->click_on){
            if (x->exc_mode == 0){ // impulse (Hann)
                float w = 0.5f - 0.5f * cosf(TWO_PI * x->click_phase);
                burst = w * x->p_click * x->note_vel;
            } else {               // DC burst (linear down)
                burst = (1.f - x->click_phase) * x->p_click * x->note_vel;
            }
            x->click_phase += x->click_inc;
            if (x->click_phase >= 1.f) x->click_on = 0;
        }

        /* noise source */
        float noise = white01(&x->rng);

        /* raw exciter before nonlinearity */
        float dry = (burst + 0.5f * noise) * x->env_level;

        /* hardness */
        float shaped = nonlin_morph(dry, x->p_hardness);

        /* precompute LP/Dry/HP branches (instant mix) */
        float lp_branch  = lp200(shaped, &x->lp200_state, x->sr);
        float dry_branch = shaped;
        float hp_branch  = hp8k  (shaped, &x->hp8k_x1, &x->hp8k_y1, x->sr);

        /* weights from Brightness + EnvDepth + Env */
        float w_lp, w_dry, w_hp;
        tilt_weights(x->p_brightness, x->env_level, x->p_envfilt, &w_lp, &w_dry, &w_hp);

        float col = w_lp*lp_branch + w_dry*dry_branch + w_hp*hp_branch;

        /* tiny stereo decorrelation (env-gated by virtue of mute above) */
        float addL =
