#include "m_pd.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/*
    exciter8~  — Universal 8‑knob stereo exciter for physical modelling
    Author: Juicy + ChatGPT
    License: MIT

    Inlets (all float 0..1):
      1 Energy         — overall drive / pressure
      2 Onset          — attack speed (0 = clicky, 1 = slow)
      3 Sustain        — hold/decay amount (0 = one‑shot, 1 = continuous)
      4 Brightness     — spectral tilt (dark ↔ bright)
      5 Hardness       — nonlinearity depth (linear→clip→tanh→fold)
      6 Turbulence     — noise vs tone/DC in the source
      7 Periodicity    — repetition/drive rate; high values add tonal driver
      8 PositionSpread — contact position color + stereo micro‑spread

    Messages:
      bang         — explicit strike (windowed impulse)
      freq <Hz>    — set tonal driver frequency (default 220 Hz)
      seed <u32>   — set noise RNG seed
      smooth <0..1>— control parameter smoothing (default ~0.005)
      hp <Hz>      — set output DC blocker cutoff (default 10 Hz)

    Audio outs:
      signal~ Left, signal~ Right

    Design notes:
      - Sources: DC (pressure), noise (turbulence), sine (tonal driver)
      - Envelope: onset/sustain controlled; also supports grainy auto retriggers
      - Nonlinearity: morph linear→clip→tanh→triangle‑fold
      - Brightness: simple tilt (LP/HP mix) to choose which modes light up
      - Position: short fractional delay color; Spread adds micro‑delay on R
      - Safety: soft clip guard + DC blocker at output
      - Universal macOS build friendly (compile with -arch x86_64 -arch arm64)
*/

// ------------------- config -------------------
#define MAX_POS_DELAY_MS   8.0f   // position coloration delay
#define MAX_SPREAD_MS      0.6f   // stereo micro delay
#define HP_DEFAULT_HZ      10.0f  // DC blocker default
#define TWO_PI             6.28318530717958647692f

#define CLAMP01(x) ((x)<0.f?0.f:((x)>1.f?1.f:(x)))
#define LERP(a,b,t) ((a)+((b)-(a))*(t))

// ------------------- type ---------------------
typedef struct _exciter8_tilde {
    t_object  x_obj;

    // signal outlets
    t_outlet *outL;
    t_outlet *outR;

    // params (targets) [0..1]
    t_float p_energy;
    t_float p_onset;
    t_float p_sustain;
    t_float p_brightness;
    t_float p_hardness;
    t_float p_turb;
    t_float p_period;
    t_float p_posspread;

    // smoothed params
    float s_energy, s_onset, s_sustain, s_brightness, s_hardness, s_turb, s_period, s_posspread;

    // smoothing
    float p_smooth;  // per‑sample smoothing coeff (0..1 small)

    // envelope state
    float env;
    float env_a_coef;   // attack coeff per sample
    float env_r_coef;   // release/decay coeff per sample
    int   env_gate;     // 0/1

    // impulse strike window
    int   hit_on;
    float hit_phase, hit_inc;   // 0..1 over the strike window
    int   hit_len;              // samples

    // periodicity / grains & tonal driver
    float grain_phase;
    float note_hz;
    float tone_phase;

    // randomness / noise
    uint32_t rng;

    // brightness filters
    float lp_state, hp_state_tmp;

    // DC blocker (highpass) states (per channel)
    float hp_a, hp_b;   // coeffs
    float hp_xL, hp_yL; // prev input/output
    float hp_xR, hp_yR;

    // position/comb buffers (stereo shared content)
    float *buf;
    int buf_len, wp; // ring buffer write index

    // sample rate
    float sr, inv_sr;

} t_exciter8_tilde;

// ------------------- helpers -------------------
static inline uint32_t xs32(uint32_t *s){
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *s = x;
}
static inline float white01(uint32_t *s){ // ~[-1,1]
    return (int32_t)xs32(s) * (1.0f/2147483648.0f);
}

static inline float smooth01(float cur, float tgt, float coeff){
    return cur + coeff * (tgt - cur);
}

static inline float tanh_fast(float x){
    const float x2 = x*x;
    return x * (27.0f + x2) / (27.0f + 9.0f*x2);
}

static inline float fold_triangle(float x){
    // triangle folding in [-1,1]
    float y = x + 1.0f;          // to [0,2]
    y -= 2.0f * floorf(y * 0.5f);// wrap [0,2) -> [0,2)
    y = fabsf(y - 1.0f);         // reflect to [0,1]
    return (y * 2.0f) - 1.0f;    // back to [-1,1]
}

static inline float nonlin_morph(float x, float hard){
    // morph linear -> clip -> tanh -> fold
    float y0 = x;
    float y1 = fmaxf(-1.f, fminf(1.f, x * (1.0f + hard*2.0f)));
    float y2 = tanh_fast(x * (1.0f + hard*3.0f));
    float y3 = fold_triangle(x * (1.0f + hard*4.0f));

    if (hard < 0.33f){
        float t = hard / 0.33f;
        return LERP(y0, y1, t);
    } else if (hard < 0.66f){
        float t = (hard - 0.33f) / 0.33f;
        return LERP(y1, y2, t);
    } else {
        float t = (hard - 0.66f) / 0.34f;
        return LERP(y2, y3, t);
    }
}

static inline int ms2samps(float ms, float sr){ return (int)(ms * 0.001f * sr); }

static inline void hp_set_coeffs(t_exciter8_tilde *x, float hz){
    // one‑pole HP derived from RC highpass: y[n]=a*(y[n-1]+in[n]-in[n-1])
    float c = expf(-TWO_PI * hz / x->sr);
    x->hp_a = c;
    x->hp_b = (1.0f + c) * 0.5f; // scale so unity at HF-ish
}

static inline float hp_proc(float in, float *x1, float *y1, float a){
    // classic one‑pole HP: y = a*(y_prev + in - x_prev)
    float y = a * ((*y1) + in - (*x1));
    *x1 = in; *y1 = y;
    return y;
}

static inline float tilt_filter(float in, float *lp, float *hp, float b, float sr){
    // b in [0..1]: 0 = dark (LP), 0.5 = flat, 1 = bright (HP)
    const float cutoff_lp = LERP(400.0f, 8000.0f, b);
    const float a_lp = expf(-2.0f * 3.1415926f * cutoff_lp / sr);
    *lp = (1.0f - a_lp) * in + a_lp * (*lp);
    float hpv = in - *lp;

    float t = (b <= 0.5f) ? (1.0f - (b*2.0f)) : 0.0f;     // LP weight when dark
    float u = (b >  0.5f) ? ((b-0.5f)*2.0f) : 0.0f;       // HP weight when bright
    float out = (t>0? (*lp)*t : 0) + (u>0? hpv*u : 0) + ((1.0f - (t+u)) * in);
    *hp = hpv;
    return out;
}

// ------------------- perform -------------------
static t_int *exciter8_tilde_perform(t_int *w){
    t_exciter8_tilde *x = (t_exciter8_tilde *)(w[1]);
    int n = (int)(w[2]);
    t_sample *outL = (t_sample *)(w[3]);
    t_sample *outR = (t_sample *)(w[4]);

    const float sr = x->sr;

    // envelope times from smoothed params (updated per block)
    float atk_ms = LERP(0.3f, 120.0f, x->s_onset);
    float rel_ms = LERP(8.0f, 1500.0f, x->s_sustain);
    x->env_a_coef = 1.0f - expf(-1.0f / (ms2samps(atk_ms, sr)+1));
    x->env_r_coef = 1.0f - expf(-1.0f / (ms2samps(rel_ms, sr)+1));

    // periodicity mapping
    float tone_mix = (x->s_period > 0.7f) ? ((x->s_period - 0.7f) / 0.3f) : 0.0f;
    if (tone_mix < 0) tone_mix = 0; if (tone_mix > 1) tone_mix = 1;
    float grain_hz = 0.f;
    if (x->s_period > 0.05f){
        float t = fminf(x->s_period, 0.7f) / 0.7f; // 0..1 in grain zone
        grain_hz = t * 40.0f; // up to ~40 Hz chatter
    }

    // position & spread (ms)
    float pos_ms = x->s_posspread * MAX_POS_DELAY_MS;
    float spread_ms = x->s_posspread * MAX_SPREAD_MS;

    // energy derivative to detect quick flicks
    static float prev_energy = 0.0f;

    for (int i = 0; i < n; ++i){
        // param smoothing per sample
        x->s_energy     = smooth01(x->s_energy,     CLAMP01(x->p_energy),     x->p_smooth);
        x->s_onset      = smooth01(x->s_onset,      CLAMP01(x->p_onset),      x->p_smooth);
        x->s_sustain    = smooth01(x->s_sustain,    CLAMP01(x->p_sustain),    x->p_smooth);
        x->s_brightness = smooth01(x->s_brightness, CLAMP01(x->p_brightness), x->p_smooth);
        x->s_hardness   = smooth01(x->s_hardness,   CLAMP01(x->p_hardness),   x->p_smooth);
        x->s_turb       = smooth01(x->s_turb,       CLAMP01(x->p_turb),       x->p_smooth);
        x->s_period     = smooth01(x->s_period,     CLAMP01(x->p_period),     x->p_smooth);
        x->s_posspread  = smooth01(x->s_posspread,  CLAMP01(x->p_posspread),  x->p_smooth);

        // edge-detect "flick" for strike
        float dE = x->s_energy - prev_energy;
        prev_energy = x->s_energy;
        if (x->s_sustain < 0.2f && dE > 0.02f){
            x->hit_on = 1;
            x->hit_phase = 0.f;
            x->hit_len = 1;       // single sample click
            x->hit_inc = 1.f;
            x->env_gate = 1;
        }

        // grains retrigger
        if (grain_hz > 0){
            float inc = grain_hz / sr;
            x->grain_phase += inc;
            if (x->grain_phase >= 1.f){
                x->grain_phase -= 1.f;
                if (x->s_sustain < 0.6f) { x->hit_on = 1; x->hit_phase = 0.f;
                    // impulse window length shaped by onset^2 (0.2..20 ms)
                    float width_ms = fmaxf(0.2f, LERP(0.2f, 20.f, x->s_onset * x->s_onset));
                    x->hit_len = ms2samps(width_ms, sr);
                    x->hit_inc = (x->hit_len > 0) ? 1.f / (float)x->hit_len : 1.f;
                }
                x->env_gate = 1;
            }
        }

        // envelope
        if (x->env_gate){
            x->env += x->env_a_coef * (x->s_energy - x->env);
            if (fabsf(x->env - x->s_energy) < 1e-4f || x->s_sustain < 0.2f) x->env_gate = 0;
        } else {
            x->env += x->env_r_coef * (0.0f - x->env);
        }

        // sources
        float noise = white01(&x->rng);          // [-1,1]
        // DC lane ~ breath/pressure
        float dc    = x->s_energy * x->s_sustain;

        // tonal driver (sine), fades in with tone_mix
        x->tone_phase += (TWO_PI) * (x->note_hz / sr);
        if (x->tone_phase > TWO_PI) x->tone_phase -= TWO_PI;
        float tone = sinf(x->tone_phase);

        // impulse (windowed tap)
        float impulse = 0.f;
        if (x->hit_on){
            float w = 0.5f - 0.5f * cosf(TWO_PI * x->hit_phase); // Hann
            impulse = w * x->s_energy;
            x->hit_phase += x->hit_inc;
            if (x->hit_phase >= 1.f) x->hit_on = 0;
        }

        // turbulence crossfade & mix
        float tonal_part = LERP(dc, tone, tone_mix);
        float src = LERP(tonal_part, noise, x->s_turb) + impulse;

        // window with envelope
        float exc = src * x->env;

        // nonlinearity (hardness) + soft safety drive
        float drive = 1.0f + x->s_hardness * 2.0f;
        float exc_nl = nonlin_morph(exc * drive, x->s_hardness);

        // brightness tilt
        float bright = tilt_filter(exc_nl, &x->lp_state, &x->hp_state_tmp, x->s_brightness, sr);

        // position/comb: write to ring buffer
        x->buf[x->wp] = bright;

        // fractional read for position color
        float pos_samps = (x->s_posspread * MAX_POS_DELAY_MS * 0.001f) * sr;
        float read_idx_f = (float)x->wp - pos_samps;
        while (read_idx_f < 0) read_idx_f += x->buf_len;
        int i0 = (int)read_idx_f;
        int i1 = (i0 + 1) % x->buf_len;
        float frac = read_idx_f - (float)i0;
        float posL = LERP(x->buf[i0], x->buf[i1], frac);

        // Right: extra micro spread delay
        float spread_samps = (x->s_posspread * MAX_SPREAD_MS * 0.001f) * sr;
        float read_idx_r = read_idx_f - spread_samps;
        while (read_idx_r < 0) read_idx_r += x->buf_len;
        int j0 = (int)read_idx_r;
        int j1 = (j0 + 1) % x->buf_len;
        float fracr = read_idx_r - (float)j0;
        float posR = LERP(x->buf[j0], x->buf[j1], fracr);

        // advance ring buffer
        x->wp = (x->wp + 1) % x->buf_len;

        // soft safety limiter (very gentle)
        const float lim = 0.98f;
        posL = tanh_fast(posL * 1.2f);
        posR = tanh_fast(posR * 1.2f);
        if (posL > lim) posL = lim; else if (posL < -lim) posL = -lim;
        if (posR > lim) posR = lim; else if (posR < -lim) posR = -lim;

        // DC blocker per channel
        float yL = hp_proc(posL, &x->hp_xL, &x->hp_yL, x->hp_a);
        float yR = hp_proc(posR, &x->hp_xR, &x->hp_yR, x->hp_a);

        // tiny dither‑like noise to avoid denormals
        float anti_denorm = 1e-10f * white01(&x->rng);
        outL[i] = yL + anti_denorm;
        outR[i] = yR - anti_denorm;
    }

    return (w + 5);
}

// ------------------- methods -------------------
static void exciter8_tilde_dsp(t_exciter8_tilde *x, t_signal **sp){
    x->sr = (float)sp[0]->s_sr;
    if (x->sr <= 0) x->sr = 44100.f;
    x->inv_sr = 1.0f / x->sr;

    // ring buffer (mono content used for stereo reads)
    float max_ms = MAX_POS_DELAY_MS + MAX_SPREAD_MS + 2.0f;
    int need = ms2samps(max_ms, x->sr) + 8;
    if (need < 512) need = 512;

    if (x->buf_len != need){
        if (x->buf) freebytes(x->buf, x->buf_len * sizeof(float));
        x->buf = (float *)getbytes(need * sizeof(float));
        memset(x->buf, 0, need*sizeof(float));
        x->buf_len = need;
        x->wp = 0;
    }

    // update HP coeffs (default if not changed by user)
    hp_set_coeffs(x, HP_DEFAULT_HZ);

    dsp_add(exciter8_tilde_perform, 4, x, sp[0]->s_n, sp[0]->s_vec, sp[1]->s_vec);
}

static void exciter8_tilde_bang(t_exciter8_tilde *x){
    // explicit strike: Hann window with length from Onset
    x->hit_on   = 1;
    x->hit_phase= 0.f;
    float width_ms = fmaxf(0.2f, LERP(0.2f, 30.f, x->p_onset * x->p_onset));
    x->hit_len  = ms2samps(width_ms, x->sr);
    x->hit_inc  = (x->hit_len > 0) ? 1.f / (float)x->hit_len : 1.f;
    x->env_gate = 1;
}

static void exciter8_tilde_freq(t_exciter8_tilde *x, t_floatarg f){
    if (f <= 0) f = 0;
    x->note_hz = f;
}

static void exciter8_tilde_seed(t_exciter8_tilde *x, t_floatarg s){
    uint32_t v = (uint32_t)(s <= 0 ? 1 : s);
    x->rng = v;
}

static void exciter8_tilde_smooth(t_exciter8_tilde *x, t_floatarg f){
    if (f < 0) f = 0;
    if (f > 1) f = 1;
    x->p_smooth = f * 0.05f; // map 0..1 UI to practical small coeff
}

static void exciter8_tilde_hp(t_exciter8_tilde *x, t_floatarg hz){
    if (hz < 1) hz = 1;
    if (hz > 60) hz = 60; // keep reasonable
    hp_set_coeffs(x, hz);
}

// ------------------- new/free -------------------
static void *exciter8_tilde_new(void){
    t_exciter8_tilde *x = (t_exciter8_tilde *)pd_new(exciter8_tilde_class);

    // defaults
    x->p_energy = x->p_onset = x->p_sustain = 0.f;
    x->p_brightness = 0.5f; // neutral tilt by default
    x->p_hardness = 0.2f;
    x->p_turb = 0.5f;
    x->p_period = 0.f;
    x->p_posspread = 0.2f;

    x->s_energy = x->s_onset = x->s_sustain = 0.f;
    x->s_brightness = 0.5f; x->s_hardness = 0.2f; x->s_turb = 0.5f; x->s_period = 0.f; x->s_posspread = 0.2f;

    x->p_smooth = 0.005f;

    x->env = 0.f; x->env_gate = 0;
    x->hit_on = 0; x->hit_phase = 0.f; x->hit_inc = 1.f; x->hit_len = 1;
    x->grain_phase = 0.f;
    x->note_hz = 220.f;
    x->tone_phase = 0.f;

    x->rng = 2220423u;
    x->lp_state = x->hp_state_tmp = 0.f;

    x->hp_a = 0.f; x->hp_b = 0.f; x->hp_xL = x->hp_yL = x->hp_xR = x->hp_yR = 0.f;

    x->buf = NULL; x->buf_len = 0; x->wp = 0;

    x->sr = sys_getsr(); if (x->sr <= 0) x->sr = 44100.f; x->inv_sr = 1.f / x->sr;

    // 8 float inlets (values write directly to fields)
    floatinlet_new(&x->x_obj, &x->p_energy);
    floatinlet_new(&x->x_obj, &x->p_onset);
    floatinlet_new(&x->x_obj, &x->p_sustain);
    floatinlet_new(&x->x_obj, &x->p_brightness);
    floatinlet_new(&x->x_obj, &x->p_hardness);
    floatinlet_new(&x->x_obj, &x->p_turb);
    floatinlet_new(&x->x_obj, &x->p_period);
    floatinlet_new(&x->x_obj, &x->p_posspread);

    // signal outlets
    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);

    return (void *)x;
}

static void exciter8_tilde_free(t_exciter8_tilde *x){
    if (x->buf) freebytes(x->buf, x->buf_len * sizeof(float));
}

// ------------------- setup -------------------
t_class *exciter8_tilde_class;

void exciter8_tilde_setup(void){
    exciter8_tilde_class = class_new(gensym("exciter8~"),
        (t_newmethod)exciter8_tilde_new,
        (t_method)exciter8_tilde_free,
        sizeof(t_exciter8_tilde), CLASS_DEFAULT, 0);

    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_dsp, gensym("dsp"), A_CANT, 0);
    class_addbang(exciter8_tilde_class, (t_method)exciter8_tilde_bang);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_freq, gensym("freq"), A_DEFFLOAT, 0);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_seed, gensym("seed"), A_DEFFLOAT, 0);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_smooth, gensym("smooth"), A_DEFFLOAT, 0);
    class_addmethod(exciter8_tilde_class, (t_method)exciter8_tilde_hp, gensym("hp"), A_DEFFLOAT, 0);
}
