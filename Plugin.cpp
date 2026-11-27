// ==============================================================================
// Model D-3 Synth  -  v2.2.0  (CPU-optimiert, schnellere Huellkurven & Filter)
// Oversampling (2x, Max-Quality Polyphase-IIR) bleibt unveraendert aktiv.
// ==============================================================================
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <memory>
#include <array>
#include <atomic>
#include <functional>

namespace
{
    // Schnelle tanh-Naeherung (Pade), exakt 1.0 bei |x| = 3
    inline float fastTanh(float x) noexcept
    {
        x = juce::jlimit(-3.0f, 3.0f, x);
        const float x2 = x * x;
        return x * (27.0f + x2) / (27.0f + 9.0f * x2);
    }

    // Praezises tanh (Kettenbruch 7. Ordnung, Fehler < 1e-5) fuer den Filter Drive
    inline float hqTanh(float x) noexcept
    {
        x = juce::jlimit(-4.97f, 4.97f, x);
        const float x2 = x * x;
        const float num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        const float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return num / den;
    }

    // sin(2*pi*p) fuer p in [0,1) - sehr billig, fuer die Sinus-Welle
    inline float fastSinTwoPi(float p) noexcept
    {
        float x = 2.0f * p;
        if (x > 1.0f) x -= 2.0f;
        const float y = 4.0f * x * (1.0f - std::abs(x));
        return 0.225f * (y * std::abs(y) - y) + y;
    }

    // Controlrate-Chunk (in Samples des Oversampling-Raums): Drift, Pitch, WT-Morph
    constexpr int kChunk = 32;
}

// ==============================================================================
// Analog Drift (Ornstein-Uhlenbeck) - laeuft jetzt nur noch mit Controlrate
// und nutzt einen billigen xorshift-Generator statt mt19937 + normal_distribution
// ==============================================================================
class AnalogDriftGenerator
{
public:
    void prepare(double controlRate, uint32_t seed)
    {
        state = (seed * 2654435761u) | 1u;
        const double twoPi = juce::MathConstants<double>::twoPi;
        slowAlpha = static_cast<float>(1.0 - std::exp(-twoPi * 0.16 / controlRate));
        fastAlpha = static_cast<float>(1.0 - std::exp(-twoPi * 2.10 / controlRate));
        slowSigma = std::sqrt(2.0f * slowAlpha) * 0.45f;
        fastSigma = std::sqrt(2.0f * fastAlpha) * 0.25f;
        slowState = noise() * 0.35f;
        fastState = 0.0f;
    }

    float next(float maxCents) noexcept
    {
        if (maxCents <= 0.0001f)
            return 0.0f;

        slowState += slowAlpha * (-slowState) + slowSigma * noise();
        fastState += fastAlpha * (-fastState) + fastSigma * noise();
        return juce::jlimit(-1.5f, 1.5f, slowState * 0.80f + fastState * 0.20f) * maxCents;
    }

private:
    // Gleichverteilt, Varianz 1
    float noise() noexcept
    {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return static_cast<float>(static_cast<int32_t>(state)) * (1.7320508f / 2147483648.0f);
    }

    uint32_t state = 1u;
    float slowAlpha = 0.0f, fastAlpha = 0.0f, slowSigma = 0.0f, fastSigma = 0.0f;
    float slowState = 0.0f, fastState = 0.0f;
};

// ==============================================================================
// Wavetable LFO (blockweise)
// ==============================================================================
class WavetableLFO
{
public:
    enum Wave { Sine = 0, Triangle, SawUp, SawDown, SampleAndHold };

    void prepare(double sampleRate)
    {
        sr = sampleRate;
        rng.seed(918273);
        uniDist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }

    void reset() { phase = 0.0; shCurrent = 0.0f; shTarget = 0.0f; }

    float processBlock(int numSamples, float rateHz, int waveType)
    {
        if (sr <= 0.0) return 0.0f;
        phase += (static_cast<double>(rateHz) * static_cast<double>(numSamples)) / sr;

        if (phase >= 1.0)
        {
            phase -= std::floor(phase);
            if (waveType == SampleAndHold)
                shTarget = uniDist(rng);
        }

        shCurrent += (shTarget - shCurrent) * 0.15f;

        const double twoPi = juce::MathConstants<double>::twoPi;
        switch (waveType)
        {
            case Sine:     return static_cast<float>(std::sin(phase * twoPi));
            case Triangle: return static_cast<float>(1.0 - 4.0 * std::abs(std::round(phase - 0.25) - (phase - 0.25)));
            case SawUp:    return static_cast<float>(2.0 * phase - 1.0);
            case SawDown:  return static_cast<float>(1.0 - 2.0 * phase);
            default:       return shCurrent;
        }
    }

private:
    double sr = 44100.0, phase = 0.0;
    float shCurrent = 0.0f, shTarget = 0.0f;
    std::mt19937 rng { 918273 };
    std::uniform_real_distribution<float> uniDist { -1.0f, 1.0f };
};

// ==============================================================================
// Analog-Oszillator: PolyBLEP/BLAMP mit weicher Reset-Flanke, gekruemmter Saege,
// Sinus-Welle und bandbegrenztem Hard-Sync (BLEP-Korrektur am Sync-Reset)
// ==============================================================================
class PolyBLEPOscillator
{
public:
    enum Waveform { Saw = 0, Triangle, Square, PulseWide, Sine };

    void reset(double initialPhase = 0.0) { phase = initialPhase - std::floor(initialPhase); pendingCorr = 0.0f; didWrap = false; }
    void setPhaseInc(double inc) noexcept { phaseInc = juce::jlimit(1.0e-4, 0.48, inc); }
    void setWaveform(int wf) noexcept { waveform = wf; }

    // Master-Info: Phase ist in diesem Sample ueberlaufen (d = Bruchteil eines Samples seit dem Reset)
    bool wrapped() const noexcept { return didWrap; }
    double wrapFrac() const noexcept { return wrapD; }

    inline float processSample() noexcept
    {
        const float out = render();
        phase += phaseInc;
        didWrap = false;
        if (phase >= 1.0) { phase -= 1.0; didWrap = true; wrapD = phase / phaseInc; }
        return out;
    }

    // Slave: wird beim Phasenueberlauf des Masters auf Phase 0 zurueckgesetzt
    inline float processSampleSync(bool masterWrapped, double d) noexcept
    {
        float out = render();
        didWrap = false;

        if (masterWrapped)
        {
            double phiR = phase + (1.0 - d) * phaseInc;          // Slave-Phase im Reset-Moment
            phiR -= std::floor(phiR);
            const float D = shape(0.0) - shape(phiR);            // Sprunghoehe
            out += 0.5f * D * static_cast<float>(d * d);         // BLEP vor dem Reset
            pendingCorr = -0.5f * D * static_cast<float>((1.0 - d) * (1.0 - d));   // BLEP nach dem Reset
            phase = d * phaseInc;
        }
        else
        {
            phase += phaseInc;
            if (phase >= 1.0) phase -= 1.0;
        }
        return out;
    }

private:
    inline float render() noexcept
    {
        const double t = phase;
        const double dt = phaseInc;
        const double dw = std::min(dt * edgeSoftness, 0.49);   // weichere Flanke = runder, "analoger"
        float out;

        switch (waveform)
        {
            case Saw:
            {
                const float x = static_cast<float>(2.0 * t - 1.0 - polyBLEP(t, dw));
                out = x + sawCurve * (x * x - 0.33333334f);    // leichte Kruemmung, DC-frei
                break;
            }
            case Square:
            case PulseWide:
            {
                const double pw = (waveform == Square) ? 0.5 : 0.25;
                double t2 = t + (1.0 - pw);
                if (t2 >= 1.0) t2 -= 1.0;
                out = ((t < pw) ? 1.0f : -1.0f) + static_cast<float>(polyBLEP(t, dw) - polyBLEP(t2, dw));
                break;
            }
            case Triangle:
            {
                double t2 = t + 0.5;
                if (t2 >= 1.0) t2 -= 1.0;
                const double tri = (t < 0.5) ? (4.0 * t - 1.0) : (3.0 - 4.0 * t);
                out = static_cast<float>(tri + 8.0 * dt * (polyBLAMP(t, dt) - polyBLAMP(t2, dt)));
                break;
            }
            default:
                out = fastSinTwoPi(static_cast<float>(t));
                break;
        }

        out += pendingCorr;
        pendingCorr = 0.0f;
        return out;
    }

    // Naive Kurvenform (ohne BLEP) fuer die Sync-Sprunghoehe
    inline float shape(double p) const noexcept
    {
        float v;
        switch (waveform)
        {
            case Saw:       { const float x = static_cast<float>(2.0 * p - 1.0); v = x + sawCurve * (x * x - 0.33333334f); break; }
            case Triangle:  v = static_cast<float>((p < 0.5) ? (4.0 * p - 1.0) : (3.0 - 4.0 * p)); break;
            case Square:    v = (p < 0.5)  ? 1.0f : -1.0f; break;
            case PulseWide: v = (p < 0.25) ? 1.0f : -1.0f; break;
            default:        v = fastSinTwoPi(static_cast<float>(p)); break;
        }
        return v;
    }

    static inline double polyBLEP(double t, double dt) noexcept
    {
        if (t < dt)       { t /= dt;            return t + t - t * t - 1.0; }
        if (t > 1.0 - dt) { t = (t - 1.0) / dt; return t * t + t + t + 1.0; }
        return 0.0;
    }

    static inline double polyBLAMP(double t, double dt) noexcept
    {
        if (t < dt)       { const double x = t / dt - 1.0;           return -(1.0 / 3.0) * x * x * x; }
        if (t > 1.0 - dt) { const double x = (t - 1.0) / dt + 1.0;   return  (1.0 / 3.0) * x * x * x; }
        return 0.0;
    }

    static constexpr double edgeSoftness = 1.35;
    static constexpr float  sawCurve = 0.07f;

    double phase = 0.0, phaseInc = 0.01, wrapD = 0.0;
    float pendingCorr = 0.0f;
    int waveform = Saw;
    bool didWrap = false;
};

// ==============================================================================
// Wavetable Data, 50 Pad-Wavetables (lazy erzeugt) + WAV Loader
// ==============================================================================
struct WavetableData
{
    static constexpr int tableSize = 2048;
    std::vector<std::vector<float>> frames;
    juce::String name { "01: Solina String Ensemble" };
};

class WavetableBank
{
public:
    static constexpr int numBuiltInPresets = 50;

    static juce::StringArray getPresetNames()
    {
        return {
            "01: Solina String Ensemble", "02: CS-80 Velvet Brass Swell", "03: OB-Xa Warm Analog Pad",
            "04: Juno-106 Chorus Silk", "05: Prophet-5 Warm Saw-Tri", "06: Jupiter-8 Lush Strings",
            "07: MiniMoog Soft Ladder Bloom", "08: PWM Creamy Ribbon Pad", "09: Dual-Detuned Analog Drift",
            "10: Warm Sub-Octave Pillow", "11: Mellotron Tape Choir", "12: Cathedral Angelic Choir",
            "13: Vowel Formant A-O-U Morph", "14: Breath & Air Silk Pad", "15: Whispering Formant Cloud",
            "16: Soprano & Alto Blend", "17: Ethereal Glass Harmonics", "18: PPG Wave 2.2 Crystal Pad",
            "19: Frozen Ice Bell Shimmer", "20: Celestial Harp Harmonics", "21: Shimmering Octave Mist",
            "22: Warm Rhodes Tine Pad", "23: DX7 Soft Sine Bloom", "24: FM Velvet Electric Pad",
            "25: FM Warm Bell Atmosphere", "26: Bowed Cello Section", "27: Chamber Strings Adagio",
            "28: Warm French Horn Swell", "29: Bass Clarinet & Woodwind", "30: Bowed Glass & Waterphone",
            "31: Hammond Warm Drawbars", "32: Pipe Organ Flutes 8'+4'", "33: Soft Reed Harmonium",
            "34: Octave Fifths Cathedral", "35: Warm Tape Saturation Morph", "36: Even-Harmonic Tube Glow",
            "37: Chebyshev Silk Overtones", "38: West-Coast Soft Wavefold", "39: Phase-Shift Warm Cloud",
            "40: Spectral Comb Breeze", "41: Golden Ratio Harmonics", "42: Fibonacci Overtones Pad",
            "43: Sub-Fifths Ambient Drone", "44: Deep Space Nebula Morph", "45: Analog Horizon Sunrise",
            "46: Arctic Aurora Shimmer", "47: Velvet Night Nocturne", "48: Liquid Silk Chorus Wave",
            "49: Dreamy Lo-Fi Tape Pad", "50: Cosmic Evolving Symphony", "51: [Custom Loaded WAV]"
        };
    }

    static std::shared_ptr<WavetableData> createBuiltInPreset(int index)
    {
        auto data = std::make_shared<WavetableData>();
        const int numFrames = 64;
        const int N = WavetableData::tableSize;
        data->frames.resize(numFrames, std::vector<float>(N, 0.0f));
        data->name = getPresetNames()[juce::jlimit(0, numBuiltInPresets - 1, index)];

        const double twoPi = juce::MathConstants<double>::twoPi;

        for (int f = 0; f < numFrames; ++f)
        {
            const double m = static_cast<double>(f) / static_cast<double>(numFrames - 1);
            float peak = 0.0001f;

            for (int i = 0; i < N; ++i)
            {
                const double w = (static_cast<double>(i) / static_cast<double>(N)) * twoPi;
                double val = 0.0;

                switch (index)
                {
                    case 0:
                        for (int h = 1; h <= 22; ++h)
                        {
                            double phaseMod = std::sin(m * twoPi + h * 0.85) * 1.4;
                            double env = std::exp(-h * (0.09 - 0.04 * m));
                            val += (env / std::pow(h, 0.7)) * std::sin(h * w + phaseMod);
                        }
                        break;
                    case 1:
                    {
                        double cutoffH = 3.0 + m * 22.0;
                        for (int h = 1; h <= 26; ++h)
                        {
                            double lp = 1.0 / (1.0 + std::pow(h / cutoffH, 3.0));
                            double res = 1.0 + 0.9 * std::exp(-std::pow((h - cutoffH * 0.85) / 2.2, 2.0));
                            val += (lp * res / h) * std::sin(h * w);
                        }
                        break;
                    }
                    case 2:
                        for (int h = 1; h <= 20; ++h)
                        {
                            double spread = m * h * 0.45;
                            val += (1.0 / std::pow(h, 0.85)) * (std::sin(h * w) + 0.65 * std::sin(h * w + spread));
                        }
                        break;
                    case 3:
                        for (int h = 1; h <= 18; ++h)
                        {
                            double c1 = std::sin(h * w);
                            double c2 = std::sin(h * w + m * 2.1 * h);
                            double c3 = std::sin(h * w - m * 1.7 * h);
                            val += (1.0 / std::pow(h, 0.8)) * (c1 + 0.55 * c2 + 0.55 * c3);
                        }
                        break;
                    case 4:
                        for (int h = 1; h <= 24; ++h)
                        {
                            double triAmp = (h % 2 == 1) ? (1.0 / (h * h)) * ((h % 4 == 1) ? 1.0 : -1.0) : 0.0;
                            double sawAmp = (1.0 / h) * std::exp(-h * 0.05);
                            val += ((1.0 - m) * triAmp * 1.4 + m * sawAmp) * std::sin(h * w);
                        }
                        break;
                    case 5:
                        for (int h = 1; h <= 24; ++h)
                        {
                            double detunePh = std::sin(m * 3.14159 * 2.0 + h * 0.5) * 1.1;
                            double rolloff = std::exp(-h * (0.08 - 0.03 * m));
                            val += (rolloff / std::pow(h, 0.75)) * (std::sin(h * w) + std::sin(h * w + detunePh));
                        }
                        break;
                    case 6:
                    {
                        double fc = 2.0 + m * 18.0;
                        for (int h = 1; h <= 24; ++h)
                        {
                            double ladder = 1.0 / std::pow(1.0 + std::pow(h / fc, 2.0), 2.0);
                            val += (ladder / std::pow(h, 0.65)) * std::sin(h * w);
                        }
                        break;
                    }
                    case 7:
                    {
                        double pw = 0.5 - 0.36 * m;
                        for (int h = 1; h <= 24; ++h)
                        {
                            double pwmAmp = (2.0 / (h * 3.14159)) * std::sin(h * 3.14159 * pw) * std::exp(-h * 0.045);
                            val += pwmAmp * std::cos(h * w);
                        }
                        break;
                    }
                    case 8:
                        for (int h = 1; h <= 20; ++h)
                        {
                            double d = m * 3.14159 * (0.3 + 0.15 * h);
                            val += (1.0 / std::pow(h, 0.8)) * (std::sin(h * w) + std::sin(h * w + d));
                        }
                        break;
                    case 9:
                        val = std::sin(w) + (0.45 + 0.25 * m) * std::sin(2.0 * w)
                            + (0.20 * m) * std::sin(3.0 * w) + (0.15 * m * m) * std::sin(4.0 * w);
                        break;
                    case 10:
                    {
                        double f1 = 3.0 + m * 2.5, f2 = 7.0 + m * 4.0;
                        for (int h = 1; h <= 16; ++h)
                        {
                            double g1 = std::exp(-std::pow((h - f1) / 1.8, 2.0));
                            double g2 = 0.65 * std::exp(-std::pow((h - f2) / 2.2, 2.0));
                            val += (0.28 / h + g1 + g2) * std::sin(h * w + 0.15 * std::sin(h * 1.7));
                        }
                        break;
                    }
                    case 11:
                    {
                        double f1 = 4.0 + std::sin(m * 3.14159) * 3.0, f2 = 10.0 + m * 5.0;
                        for (int h = 1; h <= 20; ++h)
                        {
                            double g1 = std::exp(-std::pow((h - f1) / 1.6, 2.0));
                            double g2 = 0.7 * std::exp(-std::pow((h - f2) / 2.4, 2.0));
                            val += (0.2 / h + g1 + g2) * std::sin(h * w);
                        }
                        break;
                    }
                    case 12:
                    {
                        double f1 = 6.0 - m * 3.5, f2 = 11.0 - m * 4.5;
                        for (int h = 1; h <= 18; ++h)
                        {
                            double formant = std::exp(-std::pow((h - f1) / 1.5, 2.0))
                                           + 0.7 * std::exp(-std::pow((h - f2) / 2.0, 2.0));
                            val += (0.25 / h + formant) * std::sin(h * w);
                        }
                        break;
                    }
                    case 13:
                        for (int h = 1; h <= 24; ++h)
                        {
                            double air = (h > 10) ? (m * 0.28 * std::sin(h * 2.3 + m * 4.0)) : 0.0;
                            double body = std::exp(-h * 0.16);
                            val += (body + air) * std::sin(h * w + h * 0.3);
                        }
                        break;
                    case 14:
                        for (int h = 1; h <= 20; ++h)
                        {
                            double peak1 = std::exp(-std::pow((h - (5.0 + m * 7.0)) / 2.3, 2.0));
                            val += (0.3 / h + 0.85 * peak1) * std::sin(h * w + m * h * 0.6);
                        }
                        break;
                    case 15:
                        for (int h = 1; h <= 18; ++h)
                        {
                            double alto = std::exp(-std::pow((h - 4.0) / 1.5, 2.0)) * (1.0 - 0.5 * m);
                            double sop  = std::exp(-std::pow((h - 8.0) / 2.0, 2.0)) * (0.3 + 0.7 * m);
                            val += (0.25 / h + alto + sop) * std::sin(h * w);
                        }
                        break;
                    case 16:
                        val = std::sin(w) + (0.3 + 0.4 * m) * std::sin(3.0 * w)
                            + (0.15 + 0.45 * m) * std::sin(6.0 * w) + (0.35 * m * m) * std::sin(9.0 * w);
                        break;
                    case 17:
                        for (int h = 1; h <= 16; ++h)
                        {
                            double crystal = std::sin(m * 4.5 + h * 0.9);
                            double weight = (crystal > 0.0 ? crystal : 0.0) / std::sqrt(static_cast<double>(h));
                            val += ((1.0 / (h * h)) + 0.55 * weight) * std::sin(h * w);
                        }
                        break;
                    case 18:
                        val = std::sin(w) + 0.45 * std::sin(2.0 * w) + (0.4 * m) * std::sin(5.0 * w + m * 1.5)
                            + (0.35 * m) * std::sin(7.0 * w) + (0.25 * m * m) * std::sin(11.0 * w);
                        break;
                    case 19:
                        for (int h = 1; h <= 16; ++h)
                            val += std::exp(-h * (0.24 - 0.16 * m)) * std::sin(h * w);
                        break;
                    case 20:
                        val = std::sin(w) + 0.55 * std::sin(2.0 * w) + (0.35 + 0.3 * m) * std::sin(4.0 * w)
                            + (0.45 * m) * std::sin(8.0 * w) + (0.30 * m * m) * std::sin(12.0 * w);
                        break;
                    case 21:
                        val = std::sin(w) + 0.42 * std::sin(2.0 * w)
                            + (0.22 * (1.0 - 0.4 * m)) * std::sin(3.0 * w) + (0.28 * m) * std::sin(7.0 * w);
                        break;
                    case 22:
                    {
                        double idx = 0.15 + m * 1.85;
                        val = std::sin(w + idx * std::sin(2.0 * w)) + 0.25 * std::sin(w);
                        break;
                    }
                    case 23:
                    {
                        double idx = 0.25 + m * 2.1;
                        val = std::sin(w + idx * std::sin(w) * std::cos(2.0 * w));
                        break;
                    }
                    case 24:
                    {
                        double idx = 0.2 + m * 1.9;
                        val = 0.75 * std::sin(w + idx * std::sin(3.0 * w)) + 0.35 * std::sin(2.0 * w);
                        break;
                    }
                    case 25:
                        for (int h = 1; h <= 20; ++h)
                        {
                            double bow = std::exp(-h * 0.08) * (1.0 / std::pow(h, 0.75));
                            val += bow * std::sin(h * w + m * h * 0.55);
                        }
                        break;
                    case 26:
                        for (int h = 1; h <= 18; ++h)
                        {
                            double vibr = std::sin(m * twoPi + h * 0.6) * 0.8;
                            val += (std::exp(-h * 0.11) / std::sqrt(static_cast<double>(h))) * std::sin(h * w + vibr);
                        }
                        break;
                    case 27:
                    {
                        double cutoff = 2.5 + m * 11.0;
                        for (int h = 1; h <= 16; ++h)
                        {
                            double att = 1.0 / (1.0 + std::pow(h / cutoff, 2.5));
                            val += (att / std::pow(h, 0.65)) * std::sin(h * w);
                        }
                        break;
                    }
                    case 28:
                        for (int h = 1; h <= 15; h += 2)
                        {
                            double weight = std::exp(-h * (0.14 - 0.07 * m));
                            val += (weight / std::pow(h, 0.7)) * std::sin(h * w);
                        }
                        val += 0.22 * m * std::sin(2.0 * w);
                        break;
                    case 29:
                        val = std::sin(w) + (0.5 * m) * std::sin(3.0 * w + m * 2.0)
                            + (0.35 * m) * std::sin(5.0 * w - m * 1.5) + (0.25 * m * m) * std::sin(8.0 * w);
                        break;
                    case 30:
                    {
                        const int drawbars[9] = { 1, 2, 3, 4, 6, 8, 10, 12, 16 };
                        for (int k = 0; k < 9; ++k)
                        {
                            double center = m * 6.0;
                            double wgt = std::exp(-std::pow((k - center) / 2.6, 2.0));
                            val += wgt * std::sin(drawbars[k] * w);
                        }
                        break;
                    }
                    case 31:
                        val = std::sin(w) + 0.65 * std::sin(2.0 * w) + (0.35 * m) * std::sin(3.0 * w)
                            + (0.45 * m) * std::sin(4.0 * w) + (0.20 * m * m) * std::sin(6.0 * w);
                        break;
                    case 32:
                        for (int h = 1; h <= 16; ++h)
                        {
                            double roll = std::exp(-h * (0.15 - 0.08 * m));
                            val += (roll / std::sqrt(static_cast<double>(h))) * std::sin(h * w);
                        }
                        break;
                    case 33:
                        val = std::sin(w) + 0.60 * std::sin(2.0 * w) + (0.45 + 0.25 * m) * std::sin(3.0 * w)
                            + (0.35 * m) * std::sin(4.0 * w) + (0.30 * m) * std::sin(6.0 * w);
                        break;
                    case 34:
                    {
                        double raw = std::sin(w) + 0.45 * std::sin(2.0 * w) + 0.25 * std::sin(3.0 * w);
                        double drive = 1.0 + m * 3.2;
                        val = std::tanh(raw * drive) / std::tanh(drive);
                        break;
                    }
                    case 35:
                        val = std::sin(w) + (0.2 + 0.45 * m) * std::sin(2.0 * w)
                            + (0.08 + 0.28 * m) * std::sin(4.0 * w) + (0.15 * m * m) * std::sin(6.0 * w);
                        break;
                    case 36:
                    {
                        double x = std::sin(w);
                        double t2 = 2.0 * x * x - 1.0;
                        double t3 = 4.0 * x * x * x - 3.0 * x;
                        val = x + (0.4 * m) * t2 + (0.25 * m * m) * t3;
                        break;
                    }
                    case 37:
                    {
                        double gain = 1.0 + m * 2.4;
                        val = std::sin(gain * std::sin(w));
                        break;
                    }
                    case 38:
                        for (int h = 1; h <= 14; ++h)
                        {
                            double shift = std::sin(m * twoPi + h * 0.4) * 2.0;
                            val += (std::exp(-h * 0.12) / std::sqrt(static_cast<double>(h))) * std::sin(h * w + shift);
                        }
                        break;
                    case 39:
                    {
                        double notch = 2.0 + m * 10.0;
                        for (int h = 1; h <= 18; ++h)
                        {
                            double comb = 0.5 + 0.5 * std::cos((h / notch) * 3.14159);
                            val += (comb * std::exp(-h * 0.09) / std::sqrt(static_cast<double>(h))) * std::sin(h * w);
                        }
                        break;
                    }
                    case 40:
                        val = std::sin(w) + 0.45 * std::sin(2.0 * w) + (0.4 * m) * std::sin(3.0 * w)
                            + (0.35 * m) * std::sin(5.0 * w) + (0.25 * m * m) * std::sin(8.0 * w);
                        break;
                    case 41:
                    {
                        const int fib[7] = { 1, 2, 3, 5, 8, 13, 21 };
                        for (int k = 0; k < 7; ++k)
                            val += std::exp(-k * (0.45 - 0.25 * m)) * std::sin(fib[k] * w);
                        break;
                    }
                    case 42:
                        val = std::sin(w) + 0.55 * std::sin(2.0 * w) + (0.35 + 0.25 * m) * std::sin(3.0 * w)
                            + (0.25 * m) * std::sin(6.0 * w);
                        break;
                    case 43:
                        for (int h = 1; h <= 16; ++h)
                        {
                            double swell = 0.5 + 0.5 * std::sin(m * 4.0 + h * 0.75);
                            val += (swell * std::exp(-h * 0.11) / std::sqrt(static_cast<double>(h))) * std::sin(h * w);
                        }
                        break;
                    case 44:
                        for (int h = 1; h <= 20; ++h)
                        {
                            double horizon = std::exp(-h * (0.28 - 0.22 * m));
                            val += (horizon / std::pow(h, 0.7)) * std::sin(h * w);
                        }
                        break;
                    case 45:
                        for (int h = 1; h <= 18; ++h)
                        {
                            double band = std::exp(-std::pow((h - (3.0 + m * 10.0)) / 3.0, 2.0));
                            val += (0.35 / h + 0.65 * band / std::sqrt(static_cast<double>(h))) * std::sin(h * w + h * 0.4);
                        }
                        break;
                    case 46:
                        val = std::sin(w) + (0.4 + 0.2 * m) * std::sin(2.0 * w)
                            + (0.25 * m) * std::sin(3.0 * w) + (0.12 * m) * std::sin(4.0 * w);
                        break;
                    case 47:
                        for (int h = 1; h <= 14; ++h)
                        {
                            double ph1 = std::sin(m * twoPi + h * 0.5);
                            double ph2 = std::cos(m * twoPi - h * 0.4);
                            val += (std::exp(-h * 0.13) / h) * (std::sin(h * w + ph1) + std::sin(h * w + ph2));
                        }
                        break;
                    case 48:
                        for (int h = 1; h <= 12; ++h)
                        {
                            double wobble = std::sin(h * 1.9 + m * 3.0) * 0.35;
                            val += (std::exp(-h * 0.18) / std::pow(h, 0.8)) * std::sin(h * w + wobble);
                        }
                        break;
                    default:
                        for (int h = 1; h <= 20; ++h)
                        {
                            double wave1 = 0.5 + 0.5 * std::sin(m * twoPi + h * 0.45);
                            double wave2 = 0.5 + 0.5 * std::cos(m * 3.14159 + h * 0.3);
                            val += (wave1 * wave2 * std::exp(-h * 0.08) / std::sqrt(static_cast<double>(h))) * std::sin(h * w);
                        }
                        break;
                }

                const float fVal = static_cast<float>(val);
                data->frames[f][i] = fVal;
                peak = std::max(peak, std::abs(fVal));
            }

            for (int i = 0; i < N; ++i)
                data->frames[f][i] /= peak;
        }

        return data;
    }

    static std::shared_ptr<WavetableData> loadFromWavFile(const juce::File& file)
    {
        if (!file.existsAsFile())
            return nullptr;

        juce::AudioFormatManager formatManager;
        formatManager.registerBasicFormats();

        std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
        if (reader == nullptr || reader->lengthInSamples <= 0)
            return nullptr;

        const int totalSamples = static_cast<int>(std::min<juce::int64>(reader->lengthInSamples, 2048 * 256));
        juce::AudioBuffer<float> tempBuffer(static_cast<int>(reader->numChannels), totalSamples);
        reader->read(&tempBuffer, 0, totalSamples, 0, true, true);

        std::vector<float> mono(static_cast<size_t>(totalSamples), 0.0f);
        for (int ch = 0; ch < tempBuffer.getNumChannels(); ++ch)
        {
            const float* src = tempBuffer.getReadPointer(ch);
            for (int i = 0; i < totalSamples; ++i)
                mono[static_cast<size_t>(i)] += src[i];
        }
        if (tempBuffer.getNumChannels() > 1)
        {
            const float scale = 1.0f / static_cast<float>(tempBuffer.getNumChannels());
            for (float& v : mono) v *= scale;
        }

        auto data = std::make_shared<WavetableData>();
        data->name = "WAV: " + file.getFileNameWithoutExtension();
        const int T = WavetableData::tableSize;

        if (totalSamples >= T && (totalSamples % T == 0))
        {
            const int numFrames = std::min(256, totalSamples / T);
            data->frames.resize(static_cast<size_t>(numFrames), std::vector<float>(T, 0.0f));

            for (int f = 0; f < numFrames; ++f)
            {
                const int offset = f * T;
                float peak = 0.0001f;
                for (int i = 0; i < T; ++i)
                {
                    const float val = mono[static_cast<size_t>(offset + i)];
                    data->frames[static_cast<size_t>(f)][static_cast<size_t>(i)] = val;
                    peak = std::max(peak, std::abs(val));
                }
                if (peak > 0.001f)
                    for (int i = 0; i < T; ++i)
                        data->frames[static_cast<size_t>(f)][static_cast<size_t>(i)] /= peak;
            }
        }
        else
        {
            const int numFrames = (totalSamples <= 4096) ? 1 : std::min(64, std::max(1, totalSamples / 512));
            const int chunkLen = std::max(16, totalSamples / numFrames);
            data->frames.resize(static_cast<size_t>(numFrames), std::vector<float>(T, 0.0f));

            for (int f = 0; f < numFrames; ++f)
            {
                const int start = f * chunkLen;
                const int len = std::min(chunkLen, totalSamples - start);
                if (len < 4) break;

                float peak = 0.0001f;
                for (int i = 0; i < T; ++i)
                {
                    const double pos = (static_cast<double>(i) / static_cast<double>(T)) * static_cast<double>(len - 1);
                    const int idx0 = static_cast<int>(pos);
                    const int idx1 = std::min(len - 1, idx0 + 1);
                    const float frac = static_cast<float>(pos - idx0);

                    const float val = mono[static_cast<size_t>(start + idx0)] * (1.0f - frac)
                                    + mono[static_cast<size_t>(start + idx1)] * frac;
                    data->frames[static_cast<size_t>(f)][static_cast<size_t>(i)] = val;
                    peak = std::max(peak, std::abs(val));
                }
                if (peak > 0.001f)
                    for (int i = 0; i < T; ++i)
                        data->frames[static_cast<size_t>(f)][static_cast<size_t>(i)] /= peak;
            }
        }

        if (data->frames.empty())
            return nullptr;

        return data;
    }
};

// ==============================================================================
// Wavetable-Oszillator: Frame-Auswahl wird pro Chunk vorberechnet (WtFrameRef),
// im Sample-Loop gibt es keine Locks / shared_ptr-Zugriffe mehr.
// ==============================================================================
struct WtFrameRef
{
    const float* a = nullptr;
    const float* b = nullptr;
    float frac = 0.0f;

    bool set(const WavetableData* t, float morphPos) noexcept
    {
        if (t == nullptr || t->frames.empty())
            return false;

        const int n = static_cast<int>(t->frames.size());
        const float ff = juce::jlimit(0.0f, 1.0f, morphPos) * static_cast<float>(n - 1);
        const int ia = static_cast<int>(ff);
        const int ib = std::min(n - 1, ia + 1);
        frac = ff - static_cast<float>(ia);
        a = t->frames[static_cast<size_t>(ia)].data();
        b = t->frames[static_cast<size_t>(ib)].data();
        return true;
    }
};

class WavetableOscillator
{
public:
    void reset(double initialPhase = 0.0) { phase = initialPhase - std::floor(initialPhase); }
    void setPhaseInc(double inc) noexcept { phaseInc = juce::jlimit(1.0e-4, 0.48, inc); }

    inline float processSample(const WtFrameRef& r) noexcept
    {
        const double tp = phase * static_cast<double>(WavetableData::tableSize);
        const int idx0 = static_cast<int>(tp) & (WavetableData::tableSize - 1);
        const float frac = static_cast<float>(tp - std::floor(tp));

        float out = readHermite(r.a, idx0, frac);
        if (r.b != r.a)
            out += (readHermite(r.b, idx0, frac) - out) * r.frac;

        phase += phaseInc;
        if (phase >= 1.0) phase -= 1.0;
        return out;
    }

private:
    static inline float readHermite(const float* fr, int idx, float frac) noexcept
    {
        constexpr int mask = WavetableData::tableSize - 1;
        const float ym1 = fr[(idx - 1) & mask];
        const float y0  = fr[idx & mask];
        const float y1  = fr[(idx + 1) & mask];
        const float y2  = fr[(idx + 2) & mask];

        const float c1 = 0.5f * (y1 - ym1);
        const float c2 = ym1 - 2.5f * y0 + 2.0f * y1 - 0.5f * y2;
        const float c3 = 0.5f * (y2 - ym1) + 1.5f * (y0 - y1);
        return ((c3 * frac + c2) * frac + c1) * frac + y0;
    }

    double phase = 0.0, phaseInc = 0.01;
};

// ==============================================================================
// Analoge Huellkurve (RC-artig): Attack greift sofort, Decay/Release exponentiell
// ==============================================================================
class AnalogEnvelope
{
public:
    void setSampleRate(double sr) { rate = static_cast<float>(sr); }

    void setParameters(float a, float d, float s, float r)
    {
        sustain = s;
        atkCoef = calcCoef(a * rate, atkRatio);
        atkBase = (1.0f + atkRatio) * (1.0f - atkCoef);
        decCoef = calcCoef(d * rate, drRatio);
        decBase = (sustain - drRatio) * (1.0f - decCoef);
        relCoef = calcCoef(r * rate, drRatio);
        relBase = -drRatio * (1.0f - relCoef);
    }

    void noteOn()  { stage = Attack; }
    void noteOff() { if (stage != Idle) stage = Release; }
    void reset()   { stage = Idle; level = 0.0f; }
    bool isActive() const noexcept { return stage != Idle; }
    float getLevel() const noexcept { return level; }

    inline float getNextSample() noexcept
    {
        switch (stage)
        {
            case Attack:
                level = atkBase + level * atkCoef;
                if (level >= 1.0f) { level = 1.0f; stage = Decay; }
                break;
            case Decay:
                level = decBase + level * decCoef;
                if (level <= sustain) { level = sustain; stage = Sustain; }
                break;
            case Sustain:
                level = sustain;
                break;
            case Release:
                level = relBase + level * relCoef;
                if (level <= 1.0e-4f) { level = 0.0f; stage = Idle; }
                break;
            default:
                level = 0.0f;
                break;
        }
        return level;
    }

private:
    enum Stage { Idle, Attack, Decay, Sustain, Release };

    static float calcCoef(float rateSamples, float ratio)
    {
        return rateSamples <= 1.0f ? 0.0f : std::exp(-std::log((1.0f + ratio) / ratio) / rateSamples);
    }

    static constexpr float atkRatio = 0.3f;     // kleiner = linearer, groesser = schneller Anstieg
    static constexpr float drRatio  = 1.0e-4f;

    Stage stage = Idle;
    float level = 0.0f, sustain = 1.0f, rate = 44100.0f;
    float atkCoef = 0.0f, atkBase = 1.3f, decCoef = 0.0f, decBase = 0.0f, relCoef = 0.0f, relBase = 0.0f;
};

// ==============================================================================
// 4-Pol Moog-Ladder (24 dB/Okt) - OHNE interne Cutoff-Glaettung, damit der
// Filter der Huellkurve sofort folgt. Cutoff-Koeffizient wird alle 8 Samples
// neu berechnet.
// ==============================================================================
struct MoogLadder
{
    float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;

    void reset() { s1 = s2 = s3 = s4 = 0.0f; }

    static float coeff(float cutoffHz, float rate) noexcept
    {
        const float fc = juce::jlimit(20.0f, rate * 0.25f, cutoffHz);
        const float g = std::tan(juce::MathConstants<float>::pi * fc / rate);
        return g / (1.0f + g);   // TPT-Stufengewinn G = g/(1+g)
    }

    // Zero-Delay-Feedback (TPT) 4-Pol Moog-Ladder: kein Delay im Feedback-Pfad (zero latency),
    // unkonditionell stabil bis k < 4, tanh im Loop gegen Aufschaukeln
    inline float process(float in, float g, float k) noexcept
    {
        const float v0 = (fastTanh(in - k * s4) - s1) * g;
        const float y1 = v0 + s1;  s1 = y1 + v0;
        const float v1 = (y1 - s2) * g;
        const float y2 = v1 + s2;  s2 = y2 + v1;
        const float v2 = (y2 - s3) * g;
        const float y3 = v2 + s3;  s3 = y3 + v2;
        const float v3 = (y3 - s4) * g;
        const float y4 = v3 + s4;  s4 = y4 + v3;
        // Resonanz-Kompensation: DC-Verstaerkung ist 1/(1+k) -> (1+k) haelt den Bass bei hoher Emphasis
        return y4 * (1.0f + k);
    }
};

// ==============================================================================
// Filter-Drive-Stufe (vor dem Ladder): asymmetrisches Soft-Clip -> gerade
// Harmonische (2., 4., ... = warm/fett), leichter Hoehen-Pre-Emphasis vor dem
// Clipper (Resonanzbereich sizzelt mit), Parallel-Anteil Dry/Wet (Korper bleibt
// erhalten) und DC-Blocker nach dem asymmetrischen Clip.
// ==============================================================================
struct FilterDriveStage
{
    float edgePrev = 0.0f;
    float dcInPrev = 0.0f, dcOutPrev = 0.0f;

    void reset() { edgePrev = dcInPrev = dcOutPrev = 0.0f; }

    inline float process(float x, float gain, float asym, float edge, float par, float dcR) noexcept
    {
        const float xe = x + edge * (x - edgePrev);
        edgePrev = x;

        // t + asym*x*t: der x*t-Anteil saettigt zu |x| und haelt damit die geraden
        // Harmonischen (2., 4., ...) auch bei vollem Drive erhalten (reines tanh
        // wuerde zu einem symmetrischen Rechteck kollabieren).
        const float t = hqTanh(xe * gain);
        const float wet = (t + asym * xe * t) / (hqTanh(gain) * (1.0f + asym));

        const float y = x * (1.0f - par) + wet * par;

        const float out = y - dcInPrev + dcR * dcOutPrev;
        dcInPrev = y;
        dcOutPrev = out;
        return out;
    }
};

// ==============================================================================
// Unison Sub-Voice / Poly Voice
// ==============================================================================
struct UnisonSubVoice
{
    static constexpr int maxUnison = 7;

    PolyBLEPOscillator oscs[3];
    AnalogDriftGenerator driftGens[3];
    WavetableOscillator wtOsc;
    AnalogDriftGenerator wtDriftGen;

    uint32_t noiseRng = 0x9E3779B9u;
    float pinkB0 = 0.0f, pinkB1 = 0.0f, pinkB2 = 0.0f;

    float nextNoise(bool pink)
    {
        noiseRng ^= noiseRng << 13;
        noiseRng ^= noiseRng >> 17;
        noiseRng ^= noiseRng << 5;
        const float white = static_cast<float>(static_cast<int32_t>(noiseRng)) * (1.0f / 2147483648.0f);
        if (!pink)
            return white;
        pinkB0 = 0.99765f * pinkB0 + white * 0.0990460f;
        pinkB1 = 0.96300f * pinkB1 + white * 0.2965164f;
        pinkB2 = 0.57000f * pinkB2 + white * 1.0526913f;
        return (pinkB0 + pinkB1 + pinkB2 + white * 0.1848f) * 0.05f;
    }

    void prepare(double controlRate, int polyVoiceIdx, int unisonIdx)
    {
        const uint32_t baseSeed = static_cast<uint32_t>(104729 + polyVoiceIdx * 15401 + unisonIdx * 3571);
        const double initPhaseOffset = std::fmod((polyVoiceIdx * 0.2360679 + unisonIdx * 0.38196601), 1.0);

        for (int o = 0; o < 3; ++o)
        {
            oscs[o].reset(std::fmod(initPhaseOffset + o * 0.27, 1.0));
            driftGens[o].prepare(controlRate, baseSeed + static_cast<uint32_t>(o * 1301));
        }

        wtOsc.reset(std::fmod(initPhaseOffset + 0.13, 1.0));
        wtDriftGen.prepare(controlRate, baseSeed + 4999u);

        noiseRng = baseSeed ^ 0xA5A5A5A5u ^ (static_cast<uint32_t>(unisonIdx) * 2654435761u);
        if (noiseRng == 0u) noiseRng = 0x9E3779B9u;
        pinkB0 = pinkB1 = pinkB2 = 0.0f;
    }
};

struct PolySynthVoice
{
    static constexpr int maxPolyVoices = 8;

    bool active = false;
    int midiNote = -1;
    float velocity = 1.0f;
    double currentFreq = 110.0;
    double targetFreq = 110.0;
    uint64_t noteOnAge = 0;
    uint32_t phaseRng = 0x9E3779B9u;

    double rand01() noexcept
    {
        phaseRng ^= phaseRng << 13; phaseRng ^= phaseRng >> 17; phaseRng ^= phaseRng << 5;
        return static_cast<double>(phaseRng) * (1.0 / 4294967296.0);
    }

    std::array<UnisonSubVoice, UnisonSubVoice::maxUnison> subVoices;
    AnalogEnvelope ampEnv;
    AnalogEnvelope filterEnv;
    MoogLadder ladderL, ladderR;
    FilterDriveStage fdL, fdR;

    void prepare(double osRate, double controlRate, int polyIndex)
    {
        for (int u = 0; u < UnisonSubVoice::maxUnison; ++u)
            subVoices[static_cast<size_t>(u)].prepare(controlRate, polyIndex, u);

        phaseRng = (0x9E3779B9u ^ (static_cast<uint32_t>(polyIndex + 1) * 2654435761u)) | 1u;
        ampEnv.setSampleRate(osRate);
        filterEnv.setSampleRate(osRate);
        ampEnv.reset();
        filterEnv.reset();
        ladderL.reset();
        ladderR.reset();
        fdL.reset();
        fdR.reset();
    }

    void startNote(int note, float vel, double lastGlobalFreq, bool legatoGlide, uint64_t age)
    {
        midiNote = note;
        velocity = vel;
        noteOnAge = age;
        targetFreq = juce::MidiMessage::getMidiNoteInHertz(note);

        if (!active)
        {
            ladderL.reset();
            ladderR.reset();
            fdL.reset();
            fdR.reset();
            for (auto& sub : subVoices)
            {
                for (int o = 0; o < 3; ++o) sub.oscs[o].reset(rand01());
                sub.wtOsc.reset(rand01());
            }
        }

        if (!active && !legatoGlide)
            currentFreq = targetFreq;
        else if (currentFreq <= 0.0)
            currentFreq = (lastGlobalFreq > 0.0) ? lastGlobalFreq : targetFreq;

        active = true;
        ampEnv.noteOn();
        filterEnv.noteOn();
    }

    void stopNote()
    {
        ampEnv.noteOff();
        filterEnv.noteOff();
    }
};

// ==============================================================================
// Parameter-Zeiger einmalig cachen (kein String-Lookup mehr in processBlock)
// ==============================================================================
#define MD3_PARAMS(X) \
    X(polyVoices) X(unisonVoices) X(unisonDetune) X(unisonSpread) \
    X(osc1On) X(osc1Wave) X(osc1Oct) X(osc1Detune) X(osc1Level) \
    X(osc2On) X(osc2Wave) X(osc2Oct) X(osc2Detune) X(osc2Level) \
    X(osc3On) X(osc3Wave) X(osc3Oct) X(osc3Detune) X(osc3Level) \
    X(syncMode) X(syncPitch) X(syncEnv) \
    X(wtOn) X(wtPreset) X(wtOct) X(wtDetune) X(wtPos) X(wtEnvAmt) \
    X(wtLfoWave) X(wtLfoRate) X(wtLfoAmt) X(lfoToFilter) X(wtLevel) \
    X(analogDrift) X(glide) X(drive) \
    X(cutoff) X(resonance) X(contour) X(filterDrive) X(fAttack) X(fDecay) X(fSustain) X(fRelease) \
    X(aAttack) X(aDecay) X(aSustain) X(aRelease) \
    X(arpOn) X(arpRate) X(arpMode) X(arpOctaves) X(arpGate) X(arpHold) \
    X(chorusRate) X(chorusDepth) X(chorusMix) \
    X(delayTime) X(delayFeedback) X(delayMix) X(revSize) X(revDamp) X(revMix) X(masterGain) \
    X(noiseOn) X(noiseType) X(noiseLevel) X(oversampling)

struct ParamPtrs
{
   #define X(n) std::atomic<float>* n = nullptr;
    MD3_PARAMS(X)
   #undef X

    void bind(juce::AudioProcessorValueTreeState& s)
    {
       #define X(n) n = s.getRawParameterValue(#n);
        MD3_PARAMS(X)
       #undef X
    }
};

// ==============================================================================
// AudioProcessor
// ==============================================================================
class ModelD3AudioProcessor : public juce::AudioProcessor, private juce::Timer,
                               private juce::AudioProcessorValueTreeState::Listener
{
public:
    ModelD3AudioProcessor()
        : AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true)),
          apvts(*this, nullptr, "PARAMETERS", createParameterLayout())
    {
        pp.bind(apvts);
        apvts.addParameterListener("oversampling", this);
        heldNotes.reserve(128);
        arpSorted.reserve(128);
        arpPattern.reserve(384);

        // Nur die erste Wavetable sofort erzeugen, alle anderen bei Bedarf (Message-Thread)
        publishTable(getBuiltIn(0));
        appliedPresetIdx = 0;
        startTimerHz(20);
    }

    ~ModelD3AudioProcessor() override
    {
        apvts.removeParameterListener("oversampling", this);
        stopTimer();
    }

    void parameterChanged(const juce::String& parameterID, float newValue) override
    {
        if (parameterID != "oversampling")
            return;
        juce::WeakReference<ModelD3AudioProcessor> safeThis(this);
        const int idx = juce::jlimit(0, 2, static_cast<int>(newValue + 0.5f));
        juce::MessageManager::callAsync([safeThis, idx]()
        {
            if (safeThis != nullptr)
                safeThis->applyOversampling(idx);
        });
    }

    void applyOversampling(int choiceIdx)
    {
        if (preparedSampleRate <= 0.0)
            return;
        osLog2.store(juce::jlimit(1, 3, choiceIdx + 1));

        retiredOversamplers.push_back(std::move(oversampler));
        oversampler = std::make_unique<juce::dsp::Oversampling<float>>(
            2, osLog2.load(), juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR, true);
        oversampler->initProcessing(static_cast<size_t>(preparedBlockSize));
        osPtr.store(oversampler.get());

        const double osRate = preparedSampleRate * static_cast<double>(1u << osLog2.load());
        const double controlRate = osRate / static_cast<double>(kChunk);
        for (int p = 0; p < PolySynthVoice::maxPolyVoices; ++p)
            polyVoices[static_cast<size_t>(p)].prepare(osRate, controlRate, p);
    }

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

        juce::StringArray waveChoices { "Sawtooth", "Triangle", "Square", "Pulse 25%", "Sine" };
        juce::StringArray octaveChoices { "32'", "16'", "8'", "4'", "2'" };
        juce::StringArray lfoWaveChoices { "Sine", "Triangle", "Saw Up", "Saw Down", "Smooth S&H" };
        juce::StringArray arpRateChoices { "1/4", "1/8", "1/8T", "1/16", "1/32" };
        juce::StringArray arpModeChoices { "Up", "Down", "Up/Down" };

        params.push_back(std::make_unique<juce::AudioParameterInt>("polyVoices", "Polyphony Voices", 1, 8, 8));
        params.push_back(std::make_unique<juce::AudioParameterInt>("unisonVoices", "Unison Voices", 1, 7, 4));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("unisonDetune", "Unison Detune (ct)", 0.0f, 60.0f, 14.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("unisonSpread", "Unison Stereo Spread", 0.0f, 1.0f, 0.85f));

        params.push_back(std::make_unique<juce::AudioParameterBool>("osc1On", "Osc 1 On", true));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc1Wave", "Osc 1 Wave", waveChoices, 0));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc1Oct", "Osc 1 Octave", octaveChoices, 2));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc1Detune", "Osc 1 Detune (ct)", -50.0f, 50.0f, -4.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc1Level", "Osc 1 Level", 0.0f, 1.0f, 0.65f));

        params.push_back(std::make_unique<juce::AudioParameterBool>("osc2On", "Osc 2 On", true));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc2Wave", "Osc 2 Wave", waveChoices, 0));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc2Oct", "Osc 2 Octave", octaveChoices, 2));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc2Detune", "Osc 2 Detune (ct)", -50.0f, 50.0f, 4.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc2Level", "Osc 2 Level", 0.0f, 1.0f, 0.60f));

        params.push_back(std::make_unique<juce::AudioParameterBool>("osc3On", "Osc 3 On", true));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc3Wave", "Osc 3 Wave", waveChoices, 0));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc3Oct", "Osc 3 Octave", octaveChoices, 1));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc3Detune", "Osc 3 Detune (ct)", -50.0f, 50.0f, 0.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc3Level", "Osc 3 Level", 0.0f, 1.0f, 0.50f));

        // Hard Sync: Osc 1 = Master, Osc 2/3 = Slave (Sweep ueber Sync Pitch + Filter-Huellkurve)
        params.push_back(std::make_unique<juce::AudioParameterChoice>("syncMode", "Sync Mode",
            juce::StringArray { "Off", "Osc 2 > Osc 1", "Osc 3 > Osc 1", "Osc 2+3 > Osc 1" }, 0));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("syncPitch", "Sync Pitch (st)", 0.0f, 24.0f, 7.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("syncEnv", "Sync Env Amount (st)", 0.0f, 24.0f, 0.0f));

        params.push_back(std::make_unique<juce::AudioParameterBool>("wtOn", "WT Osc On", true));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("wtPreset", "WT Preset", WavetableBank::getPresetNames(), 0));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("wtOct", "WT Osc Octave", octaveChoices, 2));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("wtDetune", "WT Osc Detune (ct)", -50.0f, 50.0f, 0.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("wtPos", "WT Position", 0.0f, 1.0f, 0.45f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("wtEnvAmt", "WT Env Mod", -1.0f, 1.0f, 0.15f));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("wtLfoWave", "WT LFO Wave", lfoWaveChoices, 0));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("wtLfoRate", "WT LFO Rate (Hz)",
            juce::NormalisableRange<float>(0.02f, 15.0f, 0.01f, 0.45f), 0.25f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("wtLfoAmt", "WT LFO Amount", -1.0f, 1.0f, 0.35f));
        params.push_back(std::make_unique<juce::AudioParameterBool>("lfoToFilter", "LFO -> Filter", false));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("wtLevel", "WT Osc Level", 0.0f, 1.0f, 0.85f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>("analogDrift", "Analog Drift (ct)", 0.0f, 25.0f, 5.5f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("glide", "Glide (ms)", 0.0f, 500.0f, 25.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("drive", "Mixer Drive", 1.0f, 6.0f, 1.6f));

        // Filter + Filter-Huellkurve (Contour jetzt in Oktaven: 0..1 = 0..5 Oktaven)
        params.push_back(std::make_unique<juce::AudioParameterFloat>("cutoff", "Filter Cutoff (Hz)",
            juce::NormalisableRange<float>(20.0f, 18000.0f, 0.1f, 0.3f), 1100.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("resonance", "Filter Emphasis", 0.0f, 1.0f, 0.22f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("contour", "Filter Contour", 0.0f, 1.0f, 0.45f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("filterDrive", "Filter Drive", 0.0f, 1.0f, 0.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("fAttack", "Filter Attack (s)",
            juce::NormalisableRange<float>(0.001f, 4.0f, 0.0001f, 0.4f), 0.004f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("fDecay", "Filter Decay (s)",
            juce::NormalisableRange<float>(0.005f, 5.0f, 0.0001f, 0.4f), 0.35f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("fSustain", "Filter Sustain", 0.0f, 1.0f, 0.30f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("fRelease", "Filter Release (s)",
            juce::NormalisableRange<float>(0.003f, 6.0f, 0.0001f, 0.4f), 0.40f));

        // Loudness-Huellkurve (schnellere Standardwerte)
        params.push_back(std::make_unique<juce::AudioParameterFloat>("aAttack", "Amp Attack (s)",
            juce::NormalisableRange<float>(0.001f, 4.0f, 0.0001f, 0.4f), 0.010f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("aDecay", "Amp Decay (s)",
            juce::NormalisableRange<float>(0.005f, 5.0f, 0.0001f, 0.4f), 0.40f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("aSustain", "Amp Sustain", 0.0f, 1.0f, 0.80f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("aRelease", "Amp Release (s)",
            juce::NormalisableRange<float>(0.003f, 6.0f, 0.0001f, 0.4f), 0.40f));

        params.push_back(std::make_unique<juce::AudioParameterBool>("arpOn", "Arpeggiator On", false));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("arpRate", "Arp Rate", arpRateChoices, 2));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("arpMode", "Arp Mode", arpModeChoices, 0));
        params.push_back(std::make_unique<juce::AudioParameterInt>("arpOctaves", "Arp Octaves", 1, 3, 2));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("arpGate", "Arp Gate", 0.1f, 0.95f, 0.75f));
        params.push_back(std::make_unique<juce::AudioParameterBool>("arpHold", "Arp Hold (Latch)", false));

        params.push_back(std::make_unique<juce::AudioParameterFloat>("chorusRate", "Chorus Rate (Hz)",
            juce::NormalisableRange<float>(0.05f, 8.0f, 0.01f, 0.5f), 0.8f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("chorusDepth", "Chorus Depth", 0.0f, 1.0f, 0.5f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("chorusMix", "Chorus Mix", 0.0f, 1.0f, 0.35f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>("delayTime", "Delay Time (ms)", 20.0f, 1200.0f, 420.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("delayFeedback", "Delay Feedback", 0.0f, 0.85f, 0.42f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("delayMix", "Delay Mix", 0.0f, 1.0f, 0.28f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>("revSize", "Reverb Size", 0.0f, 1.0f, 0.78f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("revDamp", "Reverb Damping", 0.0f, 1.0f, 0.35f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("revMix", "Reverb Mix", 0.0f, 1.0f, 0.38f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>("masterGain", "Master Volume", 0.0f, 1.5f, 0.85f));

        // Noise Oszillator (White / Pink)
        params.push_back(std::make_unique<juce::AudioParameterBool>("noiseOn", "Noise On", false));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("noiseType", "Noise Type",
            juce::StringArray { "White", "Pink" }, 0));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("noiseLevel", "Noise Level", 0.0f, 1.0f, 0.3f));

        params.push_back(std::make_unique<juce::AudioParameterChoice>("oversampling", "Oversampling",
            juce::StringArray { "2x", "4x", "8x" }, 0));

        return { params.begin(), params.end() };
    }

    // ---- Wavetable-Verwaltung (nur Message-Thread) ----------------------------
    bool loadWavetableFile(const juce::File& file)
    {
        auto newTable = WavetableBank::loadFromWavFile(file);
        if (newTable == nullptr)
            return false;

        customWavetable = newTable;
        loadedWavetablePath = file.getFullPathName();
        appliedPresetIdx = WavetableBank::numBuiltInPresets;
        publishTable(customWavetable);

        if (auto* p = apvts.getParameter("wtPreset"))
            p->setValueNotifyingHost(p->convertTo0to1(static_cast<float>(WavetableBank::numBuiltInPresets)));
        apvts.getParameter("wtOn")->setValueNotifyingHost(1.0f);
        return true;
    }

    std::shared_ptr<WavetableData> getActiveWavetable() const { return std::atomic_load(&activeWavetable); }
    float getCurrentEffectiveWtPos() const { return currentEffectiveWtPos.load(); }

    // Pegelanzeige (Peak seit der letzten Abfrage, Clip-Latch)
    float takePeakL() { return meterPeakL.exchange(0.0f); }
    float takePeakR() { return meterPeakR.exchange(0.0f); }
    bool isClipLatched() const { return clipLatched.load(); }
    void resetClip() { clipLatched.store(false); }

    const juce::String getName() const override { return "Model D-3 Synth"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 3.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Solina Velvet Ensemble"; }
    void changeProgramName(int, const juce::String&) override {}

    void prepareToPlay(double sampleRate, int samplesPerBlock) override
    {
        currentSampleRate = sampleRate;
        wtLfo.prepare(sampleRate);
        wtLfo.reset();

        preparedSampleRate = sampleRate;
        preparedBlockSize = samplesPerBlock;
        retiredOversamplers.clear();

        osLog2.store(juce::jlimit(1, 3, static_cast<int>(pp.oversampling->load()) + 1));
        oversampler = std::make_unique<juce::dsp::Oversampling<float>>(
            2, osLog2.load(), juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR, true);
        oversampler->initProcessing(static_cast<size_t>(samplesPerBlock));
        osPtr.store(oversampler.get());

        const double osRate = sampleRate * static_cast<double>(1u << osLog2.load());
        const double controlRate = osRate / static_cast<double>(kChunk);
        for (int p = 0; p < PolySynthVoice::maxPolyVoices; ++p)
            polyVoices[static_cast<size_t>(p)].prepare(osRate, controlRate, p);

        juce::dsp::ProcessSpec stereoSpec { sampleRate, static_cast<juce::uint32>(samplesPerBlock), 2 };
        juce::dsp::ProcessSpec monoSpec   { sampleRate, static_cast<juce::uint32>(samplesPerBlock), 1 };

        chorusLineL.prepare(monoSpec);
        chorusLineR.prepare(monoSpec);
        chorusLineL.setMaximumDelayInSamples(static_cast<int>(sampleRate * 0.05));
        chorusLineR.setMaximumDelayInSamples(static_cast<int>(sampleRate * 0.05));
        chorusLineL.reset();
        chorusLineR.reset();
        chorusPhase = 0.0;

        delayLineL.prepare(monoSpec);
        delayLineR.prepare(monoSpec);
        delayLineL.setMaximumDelayInSamples(static_cast<int>(sampleRate * 2.0));
        delayLineR.setMaximumDelayInSamples(static_cast<int>(sampleRate * 2.0));
        delayLineL.reset();
        delayLineR.reset();

        reverb.setSampleRate(sampleRate);
        reverb.reset();

        revSmoothK = 1.0f - std::exp(-(static_cast<double>(samplesPerBlock) / sampleRate) / 0.012);
        revSizeSm = pp.revSize->load();
        revDampSm = pp.revDamp->load();
        revMixSm  = pp.revMix->load();

        heldNotes.clear();
        arpCounter = 0.0;
        arpStep = 0;
        arpDirection = 1;
        arpNoteActive = false;
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override
    {
        return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
    }

    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override
    {
        juce::ScopedNoDenormals noDenormals;
        const int numSamples = buffer.getNumSamples();
        buffer.clear();
        if (numSamples == 0) return;

        double bpm = 120.0;
        if (auto* ph = getPlayHead())
            if (auto pos = ph->getPosition())
                if (auto hostBpm = pos->getBpm())
                    if (*hostBpm > 10.0) bpm = *hostBpm;

        const int maxAllowedPoly = juce::jlimit(1, PolySynthVoice::maxPolyVoices, static_cast<int>(pp.polyVoices->load()));
        const bool arpEnabled = pp.arpOn->load() > 0.5f;
        const bool holdNow = pp.arpHold->load() > 0.5f;
        if (arpEnabled && arpHoldWasOn && !holdNow)
            heldNotes.clear();   // Latch aus -> gelatchte Noten loslassen, Arp stoppt
        arpHoldWasOn = holdNow;

        // 1. MIDI
        for (const auto metadata : midiMessages)
        {
            const auto msg = metadata.getMessage();
            if (msg.isNoteOn())
            {
                const int note = msg.getNoteNumber();
                heldNotes.erase(std::remove(heldNotes.begin(), heldNotes.end(), note), heldNotes.end());
                heldNotes.push_back(note);
                if (!arpEnabled)
                    triggerPolyNoteOn(note, msg.getFloatVelocity(), maxAllowedPoly);
            }
            else if (msg.isNoteOff())
            {
                const int note = msg.getNoteNumber();
                const bool hold = arpEnabled && pp.arpHold->load() > 0.5f;
                if (!hold)
                    heldNotes.erase(std::remove(heldNotes.begin(), heldNotes.end(), note), heldNotes.end());
                if (!arpEnabled)
                {
                    triggerPolyNoteOff(note);
                    if (maxAllowedPoly == 1 && !heldNotes.empty())
                        triggerPolyNoteOn(heldNotes.back(), 0.85f, 1);
                }
            }
            else if (msg.isAllNotesOff() || msg.isAllSoundOff())
            {
                heldNotes.clear();
                for (auto& v : polyVoices) v.stopNote();
            }
        }

        // 2. Arpeggiator
        if (arpEnabled)
        {
            if (heldNotes.empty())
            {
                if (arpNoteActive)
                {
                    for (auto& v : polyVoices) v.stopNote();
                    arpNoteActive = false;
                }
                arpCounter = 0.0;
                arpStep = 0;
            }
            else
            {
                const int rateIdx = static_cast<int>(pp.arpRate->load());
                static const double divs[5] = { 1.0, 2.0, 3.0, 4.0, 8.0 };
                const double div = divs[juce::jlimit(0, 4, rateIdx)];

                const double samplesPerStep = (currentSampleRate * 60.0) / (bpm * div);
                const double gateRatio = pp.arpGate->load();

                arpCounter += numSamples;
                if (arpCounter >= samplesPerStep || !arpNoteActive)
                {
                    if (arpCounter >= samplesPerStep)
                        arpCounter -= samplesPerStep;

                    arpSorted = heldNotes;
                    std::sort(arpSorted.begin(), arpSorted.end());

                    const int octaves = static_cast<int>(pp.arpOctaves->load());
                    arpPattern.clear();
                    for (int o = 0; o < octaves; ++o)
                        for (int n : arpSorted)
                            arpPattern.push_back(juce::jlimit(0, 127, n + o * 12));

                    const int mode = static_cast<int>(pp.arpMode->load());
                    const int patSize = static_cast<int>(arpPattern.size());

                    if (patSize == 1)
                        arpStep = 0;
                    else if (mode == 0)
                        arpStep = (arpStep + 1) % patSize;
                    else if (mode == 1)
                    {
                        arpStep--;
                        if (arpStep < 0) arpStep = patSize - 1;
                    }
                    else
                    {
                        arpStep += arpDirection;
                        if (arpStep >= patSize) { arpStep = std::max(0, patSize - 2); arpDirection = -1; }
                        else if (arpStep < 0)   { arpStep = std::min(patSize - 1, 1); arpDirection = 1; }
                    }

                    const int trigNote = arpPattern[static_cast<size_t>(arpStep)];
                    triggerPolyNoteOn(trigNote, 0.9f, maxAllowedPoly);
                    lastArpNote = trigNote;
                    arpNoteActive = true;
                }
                else if (arpCounter > samplesPerStep * gateRatio && arpNoteActive)
                {
                    triggerPolyNoteOff(lastArpNote);
                }
            }
        }

        // 3. Globale Parameter (einmal pro Block)
        const float lfoRaw = wtLfo.processBlock(numSamples, pp.wtLfoRate->load(),
                                                static_cast<int>(pp.wtLfoWave->load()))
                           * pp.wtLfoAmt->load() * 0.5f;
        const bool lfoToFilterOn = pp.lfoToFilter->load() > 0.5f;
        const float lfoModVal = lfoToFilterOn ? 0.0f : lfoRaw;          // nur Wavetable
        const float lfoFilt   = lfoToFilterOn ? lfoRaw * 2.0f : 0.0f;   // nur Filter (+-2 Oktaven)
        const float lfoFiltStart = lfoFiltPrev;
        lfoFiltPrev = lfoFilt;

        const float aA = pp.aAttack->load(), aD = pp.aDecay->load(), aS = pp.aSustain->load(), aR = pp.aRelease->load();
        const float fA = pp.fAttack->load(), fD = pp.fDecay->load(), fS = pp.fSustain->load(), fR = pp.fRelease->load();

        const int numUnison = juce::jlimit(1, UnisonSubVoice::maxUnison, static_cast<int>(pp.unisonVoices->load()));
        const double maxUnisonDetune = static_cast<double>(pp.unisonDetune->load());
        const float unisonSpread = pp.unisonSpread->load();

        std::array<double, UnisonSubVoice::maxUnison> voiceDetuneCents {};
        std::array<float, UnisonSubVoice::maxUnison> voicePanL {};
        std::array<float, UnisonSubVoice::maxUnison> voicePanR {};

        if (numUnison == 1)
        {
            voicePanL[0] = 0.70710678f;
            voicePanR[0] = 0.70710678f;
        }
        else
        {
            float gSq = 0.0f;
            for (int u = 0; u < numUnison; ++u)
            {
                const float np = (2.0f * static_cast<float>(u) / static_cast<float>(numUnison - 1)) - 1.0f;
                const float gu = 1.0f - 0.30f * std::abs(np);
                gSq += gu * gu;
            }
            const float gainNorm = std::sqrt(static_cast<float>(numUnison) / gSq);

            for (int u = 0; u < numUnison; ++u)
            {
                const double normPos = (2.0 * u / static_cast<double>(numUnison - 1)) - 1.0;
                const double curved = std::copysign(std::pow(std::abs(normPos), 1.15), normPos);
                voiceDetuneCents[static_cast<size_t>(u)] = curved * maxUnisonDetune;

                const float panAngle = static_cast<float>((normPos * unisonSpread + 1.0) * 0.25 * juce::MathConstants<double>::pi);
                const float gu = (1.0f - 0.30f * std::abs(static_cast<float>(normPos))) * gainNorm;
                voicePanL[static_cast<size_t>(u)] = std::cos(panAngle) * gu;
                voicePanR[static_cast<size_t>(u)] = std::sin(panAngle) * gu;
            }
        }

        const float unisonGainComp = 1.0f / std::sqrt(static_cast<float>(numUnison));
        const float polyGainComp = 0.52f;

        const float oscLevel[3] = { pp.osc1Level->load(), pp.osc2Level->load(), pp.osc3Level->load() };
        const bool oscActive[3] = {
            pp.osc1On->load() > 0.5f && oscLevel[0] > 0.0005f,
            pp.osc2On->load() > 0.5f && oscLevel[1] > 0.0005f,
            pp.osc3On->load() > 0.5f && oscLevel[2] > 0.0005f };
        const int oscWave[3] = { static_cast<int>(pp.osc1Wave->load()), static_cast<int>(pp.osc2Wave->load()), static_cast<int>(pp.osc3Wave->load()) };
        const int oscOct[3]  = { static_cast<int>(pp.osc1Oct->load()),  static_cast<int>(pp.osc2Oct->load()),  static_cast<int>(pp.osc3Oct->load()) };
        const float oscDetune[3] = { pp.osc1Detune->load(), pp.osc2Detune->load(), pp.osc3Detune->load() };

        const float noiseLevel = pp.noiseLevel->load();
        const bool noiseActive = pp.noiseOn->load() > 0.5f && noiseLevel > 0.0005f;
        const bool noisePink = static_cast<int>(pp.noiseType->load()) == 1;

        const int syncModeIdx = static_cast<int>(pp.syncMode->load());
        const bool slaveSync[3] = { false, syncModeIdx == 1 || syncModeIdx == 3, syncModeIdx == 2 || syncModeIdx == 3 };
        const bool syncAny = syncModeIdx != 0;
        const double syncPitchSt = static_cast<double>(pp.syncPitch->load());
        const double syncEnvSt = static_cast<double>(pp.syncEnv->load());

        const float wtLevel = pp.wtLevel->load();
        const bool wtOn = pp.wtOn->load() > 0.5f && wtLevel > 0.0005f;
        const int wtOct = static_cast<int>(pp.wtOct->load());
        const float wtDet = pp.wtDetune->load();
        const float wtPos = pp.wtPos->load();
        const float wtEnv = pp.wtEnvAmt->load();

        static const double octMults[5] = { 0.25, 0.5, 1.0, 2.0, 4.0 };
        const double osRate = currentSampleRate * static_cast<double>(1u << osLog2.load());
        const float glideMs = pp.glide->load();
        const double glideChunkCoeff = (glideMs <= 0.5f) ? 1.0
            : (1.0 - std::exp(-static_cast<double>(kChunk) / (osRate * (glideMs * 0.001))));
        const float driftCents = pp.analogDrift->load();
        static const float oscDriftScale[3] = { 0.85f, 1.05f, 1.20f };

        const float drive = pp.drive->load();
        const float driveNorm = 1.0f / fastTanh(drive);
        const float mixBias = 0.12f, mixBiasOut = fastTanh(0.12f);
        const float baseCutoff = pp.cutoff->load();
        // k >= 4 -> echte Selbstoszillation ("Zwitschern"); tanh im ZDF-Loop haelt es begrenzt.
        // (1+k)-Kompensation am Filterausgang gleicht den Bassverlust wieder aus.
        const float resK = pp.resonance->load() * 4.15f;
        const float contourOct = pp.contour->load() * 5.0f;

        // Filter Drive (vor dem Ladder): 0..1 = 0..+30 dB Gain in asymmetrisches Soft-Clip
        // (gerade Harmonische = fett), peak-normiert. Asymmetrie/Edge/Parallel-Anteil folgen
        // dem Regler, DC-Blocker-Pol aus der Oversampling-Rate (siehe FilterDriveStage).
        const float fDrive = pp.filterDrive->load();
        const bool fdOn = fDrive > 0.001f;
        const float fdGain = std::pow(10.0f, fDrive * (30.0f / 20.0f));
        const float fdAsym = 0.80f * fDrive;
        const float fdEdge = 0.30f * fDrive;
        const float fdPar  = 0.35f + 0.30f * fDrive;
        const float fdDcR  = std::exp(-juce::MathConstants<float>::twoPi * 20.0f / static_cast<float>(osRate));

        // Wavetable einmal pro Block holen (kein Lock im Sample-Loop)
        const auto table = std::atomic_load(&activeWavetable);
        const WavetableData* tablePtr = table.get();

        bool anyVoiceActive = false;
        for (auto& pv : polyVoices) anyVoiceActive = anyVoiceActive || pv.active;

        float latestWtPosForGui = juce::jlimit(0.0f, 1.0f, wtPos + lfoModVal);

        // 4. Rendern im 2x Oversampling-Raum
        juce::dsp::AudioBlock<float> stereoBlock(buffer);

        if (anyVoiceActive)
        {
            auto osBlock = osPtr.load()->processSamplesUp(stereoBlock);
            const int osN = static_cast<int>(osBlock.getNumSamples());
            const float invOsN = 1.0f / static_cast<float>(osN);
            float* outL = osBlock.getChannelPointer(0);
            float* outR = osBlock.getChannelPointer(1);

            for (auto& pv : polyVoices)
            {
                if (!pv.active) continue;

                pv.ampEnv.setParameters(std::max(aA, 0.0012f), aD, aS, aR);
                pv.filterEnv.setParameters(fA, fD, fS, fR);

                for (int u = 0; u < numUnison; ++u)
                    for (int o = 0; o < 3; ++o)
                        pv.subVoices[static_cast<size_t>(u)].oscs[o].setWaveform(oscWave[o]);

                float g = 0.0f;
                WtFrameRef wtRef;

                for (int c0 = 0; c0 < osN; c0 += kChunk)
                {
                    const int cn = std::min(kChunk, osN - c0);

                    // --- Controlrate: Glide, Drift, Pitch, Wavetable-Position ---
                    pv.currentFreq += (pv.targetFreq - pv.currentFreq) * glideChunkCoeff;

                    const float effPos = juce::jlimit(0.0f, 1.0f, wtPos + pv.filterEnv.getLevel() * wtEnv + lfoModVal);
                    latestWtPosForGui = effPos;
                    const bool wtActive = wtOn && wtRef.set(tablePtr, effPos);
                    const double syncSemis = syncPitchSt + static_cast<double>(pv.filterEnv.getLevel()) * syncEnvSt;

                    for (int u = 0; u < numUnison; ++u)
                    {
                        auto& sub = pv.subVoices[static_cast<size_t>(u)];
                        const double uDet = voiceDetuneCents[static_cast<size_t>(u)];

                        for (int o = 0; o < 3; ++o)
                        {
                            if (!oscActive[o] && !(o == 0 && syncAny)) continue;
                            const double cents = static_cast<double>(oscDetune[o]) + uDet
                                               + sub.driftGens[o].next(driftCents * oscDriftScale[o])
                                               + (slaveSync[o] ? syncSemis * 100.0 : 0.0);
                            sub.oscs[o].setPhaseInc(pv.currentFreq * octMults[oscOct[o]] * std::exp2(cents / 1200.0) / osRate);
                        }

                        if (wtActive)
                        {
                            const double cents = static_cast<double>(wtDet) + uDet + sub.wtDriftGen.next(driftCents * 0.5f);
                            sub.wtOsc.setPhaseInc(pv.currentFreq * octMults[wtOct] * std::exp2(cents / 1200.0) / osRate);
                        }
                    }

                    // --- Samplerate: Oszillatoren, Sattigung, Filter, VCA ---
                    for (int s = c0; s < c0 + cn; ++s)
                    {
                        float sumL = 0.0f, sumR = 0.0f;

                        for (int u = 0; u < numUnison; ++u)
                        {
                            auto& sub = pv.subVoices[static_cast<size_t>(u)];
                            float x = 0.0f;

                            if (oscActive[0] || syncAny)
                            {
                                const float m = sub.oscs[0].processSample();
                                if (oscActive[0]) x += m * oscLevel[0];
                            }

                            for (int o = 1; o < 3; ++o)
                                if (oscActive[o])
                                    x += (slaveSync[o] ? sub.oscs[o].processSampleSync(sub.oscs[0].wrapped(), sub.oscs[0].wrapFrac())
                                                       : sub.oscs[o].processSample()) * oscLevel[o];

                            if (wtActive)
                                x += sub.wtOsc.processSample(wtRef) * wtLevel;

                            if (noiseActive)
                                x += sub.nextNoise(noisePink) * noiseLevel;

                            sumL += x * voicePanL[static_cast<size_t>(u)];
                            sumR += x * voicePanR[static_cast<size_t>(u)];
                        }

                        const float envA = pv.ampEnv.getNextSample() * pv.velocity;
                        const float envF = pv.filterEnv.getNextSample();

                        // Cutoff folgt Huellkurve + LFO in jedem Sample (maximal knackig, zero-latency)
                        const float lfoFiltNow = lfoFiltStart + (lfoFilt - lfoFiltStart) * (static_cast<float>(s) * invOsN);
                        g = MoogLadder::coeff(baseCutoff * std::exp2(envF * contourOct + lfoFiltNow), static_cast<float>(osRate));

                        float inL = (fastTanh(sumL * unisonGainComp * drive + mixBias) - mixBiasOut) * driveNorm;
                        float inR = (fastTanh(sumR * unisonGainComp * drive + mixBias) - mixBiasOut) * driveNorm;

                        if (fdOn)
                        {
                            inL = pv.fdL.process(inL, fdGain, fdAsym, fdEdge, fdPar, fdDcR);
                            inR = pv.fdR.process(inR, fdGain, fdAsym, fdEdge, fdPar, fdDcR);
                        }

                        // Moog-Signalweg: Mixer -> Filter Drive -> VCF -> VCA
                        outL[s] += pv.ladderL.process(inL, g, resK) * envA * polyGainComp;
                        outR[s] += pv.ladderR.process(inR, g, resK) * envA * polyGainComp;
                    }
                }

                if (!pv.ampEnv.isActive())
                    pv.active = false;
            }

            osPtr.load()->processSamplesDown(stereoBlock);
        }

        currentEffectiveWtPos.store(latestWtPosForGui);

        auto* leftPtr  = buffer.getWritePointer(0);
        auto* rightPtr = buffer.getWritePointer(1);

        // 5. Stereo Chorus (LFO-modulierte Delaylines, gegenphasig L/R)
        const float chorusMix = pp.chorusMix->load();
        if (chorusMix > 0.001f)
        {
            const float chorusRate  = pp.chorusRate->load();
            const float chorusDepth = pp.chorusDepth->load();
            const float centerS = 18.0f * 0.001f * currentSampleRate;
            const float modS    = 9.0f * 0.001f * currentSampleRate * chorusDepth;
            const double phaseInc = static_cast<double>(chorusRate) / currentSampleRate;

            for (int i = 0; i < numSamples; ++i)
            {
                const double lfo = std::sin(juce::MathConstants<double>::twoPi * chorusPhase);
                chorusPhase += phaseInc;
                if (chorusPhase >= 1.0)
                    chorusPhase -= 1.0;

                const float inL = leftPtr[i];
                const float inR = rightPtr[i];

                const float dL = juce::jmax(1.0f, centerS + static_cast<float>(lfo) * modS);
                const float dR = juce::jmax(1.0f, centerS - static_cast<float>(lfo) * modS);

                const float wetL = chorusLineL.popSample(0, dL, true);
                const float wetR = chorusLineR.popSample(0, dR, true);
                chorusLineL.pushSample(0, inL);
                chorusLineR.pushSample(0, inR);

                leftPtr[i]  = inL * (1.0f - chorusMix * 0.5f) + wetL * chorusMix;
                rightPtr[i] = inR * (1.0f - chorusMix * 0.5f) + wetR * chorusMix;
            }
        }

        // 6. Stereo Cross-Feedback Delay
        const float delayMix = pp.delayMix->load();
        if (delayMix > 0.001f)
        {
            const float delayMs = pp.delayTime->load();
            const float delayFb = pp.delayFeedback->load();
            const float dSamplesL = static_cast<float>((delayMs * 0.001) * currentSampleRate);
            const float dSamplesR = static_cast<float>((delayMs * 0.001 * 1.12) * currentSampleRate);

            for (int i = 0; i < numSamples; ++i)
            {
                const float inL = leftPtr[i], inR = rightPtr[i];
                const float delL = delayLineL.popSample(0, dSamplesL, true);
                const float delR = delayLineR.popSample(0, dSamplesR, true);

                delayLineL.pushSample(0, inL + delR * delayFb);
                delayLineR.pushSample(0, inR + delL * delayFb);

                leftPtr[i]  = inL * (1.0f - delayMix * 0.5f) + delL * delayMix;
                rightPtr[i] = inR * (1.0f - delayMix * 0.5f) + delR * delayMix;
            }
        }

        // 7. Reverb (Parameter pro Block geglaettet -> kein Knistern beim Drehen)
        revSizeSm += (pp.revSize->load() - revSizeSm) * revSmoothK;
        revDampSm += (pp.revDamp->load() - revDampSm) * revSmoothK;
        revMixSm  += (pp.revMix->load()  - revMixSm)  * revSmoothK;

        if (revMixSm > 0.001f)
        {
            juce::Reverb::Parameters rp;
            rp.roomSize = revSizeSm;
            rp.damping  = revDampSm;
            rp.wetLevel = revMixSm * 0.85f;
            rp.dryLevel = 1.0f - (revMixSm * 0.35f);
            rp.width    = 1.0f;
            reverb.setParameters(rp);
            reverb.processStereo(leftPtr, rightPtr, numSamples);
        }

        // 8. Master + Safety-Limiter (Soft-Clip) mit Pegelmessung vor dem Clipper
        const float master = pp.masterGain->load();
        float peaks[2] = { 0.0f, 0.0f };
        for (int ch = 0; ch < 2; ++ch)
        {
            auto* d = buffer.getWritePointer(ch);
            float pk = 0.0f;
            for (int i = 0; i < numSamples; ++i)
            {
                const float v = d[i] * master;
                pk = std::max(pk, std::abs(v));
                d[i] = fastTanh(v);
            }
            peaks[ch] = pk;
        }
        if (peaks[0] > meterPeakL.load()) meterPeakL.store(peaks[0]);
        if (peaks[1] > meterPeakR.load()) meterPeakR.store(peaks[1]);
        if (std::max(peaks[0], peaks[1]) > 1.0f) clipLatched.store(true);
    }

    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;

    void getStateInformation(juce::MemoryBlock& destData) override
    {
        auto state = apvts.copyState();
        state.setProperty("loadedWavetablePath", loadedWavetablePath, nullptr);
        std::unique_ptr<juce::XmlElement> xml(state.createXml());
        copyXmlToBinary(*xml, destData);
    }

    void setStateInformation(const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
        if (xmlState != nullptr && xmlState->hasTagName(apvts.state.getType()))
        {
            auto tree = juce::ValueTree::fromXml(*xmlState);
            const juce::String path = tree.getProperty("loadedWavetablePath", "").toString();
            apvts.replaceState(tree);
            if (path.isNotEmpty())
                loadWavetableFile(juce::File(path));
        }
    }

    juce::AudioProcessorValueTreeState apvts;

private:
    // Preset-Wechsel laeuft ueber den Message-Thread (Timer) - die Tabelle wird bei
    // Bedarf erzeugt, nie im Audio-Thread.
    void timerCallback() override
    {
        const int idx = static_cast<int>(pp.wtPreset->load());
        if (idx == appliedPresetIdx)
            return;

        if (idx < WavetableBank::numBuiltInPresets)
        {
            customWavetable.reset();
            loadedWavetablePath.clear();
            publishTable(getBuiltIn(idx));
        }
        else if (customWavetable != nullptr)
        {
            publishTable(customWavetable);
        }
        appliedPresetIdx = idx;
    }

    std::shared_ptr<WavetableData> getBuiltIn(int idx)
    {
        auto& slot = builtInTables[static_cast<size_t>(idx)];
        if (slot == nullptr)
            slot = WavetableBank::createBuiltInPreset(idx);
        return slot;
    }

    void publishTable(std::shared_ptr<WavetableData> table)
    {
        if (table == nullptr) return;
        auto old = std::atomic_exchange(&activeWavetable, table);
        if (old != nullptr)
        {
            // Alte Tabelle am Leben halten, damit sie nie im Audio-Thread freigegeben wird
            retiredTables.push_back(std::move(old));
            if (retiredTables.size() > 16)
                retiredTables.erase(retiredTables.begin());
        }
    }

    void triggerPolyNoteOn(int midiNote, float velocity, int maxAllowedPoly)
    {
        globalNoteCounter++;
        const double newFreq = juce::MidiMessage::getMidiNoteInHertz(midiNote);

        for (int i = maxAllowedPoly; i < PolySynthVoice::maxPolyVoices; ++i)
            polyVoices[static_cast<size_t>(i)].active = false;

        for (int i = 0; i < maxAllowedPoly; ++i)
        {
            auto& v = polyVoices[static_cast<size_t>(i)];
            if (v.active && v.midiNote == midiNote)
            {
                v.startNote(midiNote, velocity, lastTriggeredFreq, maxAllowedPoly == 1, globalNoteCounter);
                lastTriggeredFreq = newFreq;
                return;
            }
        }

        for (int i = 0; i < maxAllowedPoly; ++i)
        {
            auto& v = polyVoices[static_cast<size_t>(i)];
            if (!v.active)
            {
                v.startNote(midiNote, velocity, lastTriggeredFreq, maxAllowedPoly == 1, globalNoteCounter);
                lastTriggeredFreq = newFreq;
                return;
            }
        }

        int oldestIdx = 0;
        uint64_t oldestAge = polyVoices[0].noteOnAge;
        for (int i = 1; i < maxAllowedPoly; ++i)
            if (polyVoices[static_cast<size_t>(i)].noteOnAge < oldestAge)
            {
                oldestAge = polyVoices[static_cast<size_t>(i)].noteOnAge;
                oldestIdx = i;
            }

        polyVoices[static_cast<size_t>(oldestIdx)].startNote(midiNote, velocity, lastTriggeredFreq,
                                                             maxAllowedPoly == 1, globalNoteCounter);
        lastTriggeredFreq = newFreq;
    }

    void triggerPolyNoteOff(int midiNote)
    {
        for (auto& pv : polyVoices)
            if (pv.active && pv.midiNote == midiNote)
                pv.stopNote();
    }

    ParamPtrs pp;

    double currentSampleRate = 44100.0;
    std::atomic<int> osLog2 { 1 };
    std::atomic<juce::dsp::Oversampling<float>*> osPtr { nullptr };
    double preparedSampleRate = 0.0;
    int preparedBlockSize = 0;
    std::vector<std::unique_ptr<juce::dsp::Oversampling<float>>> retiredOversamplers;
    double lastTriggeredFreq = 110.0;
    uint64_t globalNoteCounter = 0;

    WavetableLFO wtLfo;
    std::array<PolySynthVoice, PolySynthVoice::maxPolyVoices> polyVoices;

    std::array<std::shared_ptr<WavetableData>, WavetableBank::numBuiltInPresets> builtInTables;
    std::shared_ptr<WavetableData> customWavetable;
    std::shared_ptr<WavetableData> activeWavetable;
    std::vector<std::shared_ptr<WavetableData>> retiredTables;
    juce::String loadedWavetablePath;
    int appliedPresetIdx = 0;
    std::atomic<float> currentEffectiveWtPos { 0.45f };
    std::atomic<float> meterPeakL { 0.0f }, meterPeakR { 0.0f };
    std::atomic<bool> clipLatched { false };

    std::unique_ptr<juce::dsp::Oversampling<float>> oversampler;
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> chorusLineL;
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> chorusLineR;
    double chorusPhase = 0.0;
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> delayLineL;
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> delayLineR;
    juce::Reverb reverb;
    float revSizeSm = 0.78f, revDampSm = 0.35f, revMixSm = 0.38f;
    float revSmoothK = 0.0f;
    float lfoFiltPrev = 0.0f;

    std::vector<int> heldNotes, arpSorted, arpPattern;
    double arpCounter = 0.0;
    int arpStep = 0;
    int arpDirection = 1;
    bool arpNoteActive = false;
    bool arpHoldWasOn = false;
    int lastArpNote = 60;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModelD3AudioProcessor)
    JUCE_DECLARE_WEAK_REFERENCEABLE(ModelD3AudioProcessor)
};

// ==============================================================================
// 2D Wavetable Visualizer (zeichnet nur noch, wenn sich etwas geaendert hat)
// ==============================================================================
class Wavetable2DDisplay : public juce::Component, private juce::Timer
{
public:
    explicit Wavetable2DDisplay(ModelD3AudioProcessor& p) : processor(p) { startTimerHz(24); }

    void paint(juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();
        g.setColour(juce::Colour(0xff101318));
        g.fillRoundedRectangle(bounds, 6.0f);
        g.setColour(juce::Colour(0xff2e3642));
        g.drawRoundedRectangle(bounds.reduced(0.5f), 6.0f, 1.0f);

        auto area = bounds.reduced(10.0f, 8.0f);
        auto headerArea = area.removeFromTop(18.0f);

        auto table = processor.getActiveWavetable();
        const float morphPos = processor.getCurrentEffectiveWtPos();

        g.setColour(juce::Colour(0xff8fa3bf));
        g.setFont(juce::Font(juce::FontOptions(12.0f, juce::Font::bold)));
        g.drawText((table != nullptr) ? table->name : juce::String("Keine Wavetable"),
                   headerArea.removeFromLeft(headerArea.getWidth() * 0.72f), juce::Justification::centredLeft, true);

        g.setColour(juce::Colour(0xff4da6ff));
        g.drawText("Pos: " + juce::String(static_cast<int>(morphPos * 100.0f)) + "%", headerArea,
                   juce::Justification::centredRight, false);

        if (table == nullptr || table->frames.empty())
            return;

        const int numFrames = static_cast<int>(table->frames.size());
        const int activeFrameIdx = juce::jlimit(0, numFrames - 1, static_cast<int>(morphPos * static_cast<float>(numFrames - 1)));

        g.setColour(juce::Colour(0xff1f2630));
        g.drawHorizontalLine(static_cast<int>(area.getCentreY()), area.getX(), area.getRight());

        const int step = std::max(1, numFrames / 16);
        for (int f = 0; f < numFrames; f += step)
        {
            if (f == activeFrameIdx) continue;
            const float dist = std::abs(static_cast<float>(f - activeFrameIdx)) / static_cast<float>(std::max(1, numFrames));
            const float alpha = juce::jlimit(0.06f, 0.24f, 0.26f * (1.0f - dist));
            drawWaveformPath(g, table->frames[static_cast<size_t>(f)], area,
                             juce::Colour(0xff4da6ff).withAlpha(alpha), 1.0f, (f - activeFrameIdx) * 0.12f);
        }

        drawWaveformPath(g, table->frames[static_cast<size_t>(activeFrameIdx)], area, juce::Colour(0xffffaa33), 2.2f, 0.0f);
    }

private:
    static void drawWaveformPath(juce::Graphics& g, const std::vector<float>& frame,
                                 juce::Rectangle<float> area, juce::Colour col, float thickness, float yShift)
    {
        if (frame.empty()) return;
        juce::Path path;
        const int numPoints = 256;
        const int tableLen = static_cast<int>(frame.size());

        for (int i = 0; i < numPoints; ++i)
        {
            const float normX = static_cast<float>(i) / static_cast<float>(numPoints - 1);
            const int idx = juce::jlimit(0, tableLen - 1, static_cast<int>(normX * static_cast<float>(tableLen - 1)));
            const float sample = juce::jlimit(-1.05f, 1.05f, frame[static_cast<size_t>(idx)]);

            const float x = area.getX() + normX * area.getWidth();
            const float y = area.getCentreY() - sample * (area.getHeight() * 0.44f) + yShift;

            if (i == 0) path.startNewSubPath(x, y);
            else        path.lineTo(x, y);
        }

        g.setColour(col);
        g.strokePath(path, juce::PathStrokeType(thickness));
    }

    void timerCallback() override
    {
        auto t = processor.getActiveWavetable();
        const float m = processor.getCurrentEffectiveWtPos();
        if (t.get() != lastTable || std::abs(m - lastMorph) > 0.002f)
        {
            lastTable = t.get();
            lastMorph = m;
            repaint();
        }
    }

    ModelD3AudioProcessor& processor;
    const WavetableData* lastTable = nullptr;
    float lastMorph = -1.0f;
};

// ==============================================================================
// Pegelanzeige (Stereo, Peak-Hold, CLIP-LED - Klick auf die Anzeige setzt CLIP zurueck)
// ==============================================================================
class LevelMeter : public juce::Component, private juce::Timer
{
public:
    explicit LevelMeter(ModelD3AudioProcessor& p) : processor(p)
    {
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
        startTimerHz(30);
    }

    void mouseDown(const juce::MouseEvent&) override
    {
        processor.resetClip();
        clip = false;
        holdL = holdR = minDb;
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        auto b = getLocalBounds().toFloat();

        auto led = b.removeFromRight(44.0f).withSizeKeepingCentre(40.0f, 16.0f);
        g.setColour(clip ? juce::Colour(0xffff2a2a) : juce::Colour(0xff3a2226));
        g.fillRoundedRectangle(led, 3.0f);
        g.setColour(clip ? juce::Colours::white : juce::Colour(0xff7a5a5e));
        g.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
        g.drawText("CLIP", led, juce::Justification::centred, false);

        auto labelArea = b.removeFromLeft(30.0f);
        g.setColour(juce::Colour(0xff8fa3bf));
        g.drawText("OUT", labelArea.withHeight(23.0f), juce::Justification::centredLeft, false);

        b.removeFromLeft(4.0f);
        b.removeFromRight(8.0f);

        auto barL = b.removeFromTop(10.0f);
        b.removeFromTop(3.0f);
        auto barR = b.removeFromTop(10.0f);
        b.removeFromTop(1.0f);
        drawBar(g, barL, dispL, holdL);
        drawBar(g, barR, dispR, holdR);

        g.setColour(juce::Colour(0xff8fa3bf));
        g.setFont(juce::Font(juce::FontOptions(8.0f)));
        for (float db : { -48.0f, -24.0f, -12.0f, -6.0f, 0.0f })
            g.drawText(juce::String(static_cast<int>(db)),
                       juce::Rectangle<float>(dbToX(db, barL) - 14.0f, b.getY(), 28.0f, b.getHeight()),
                       juce::Justification::centred, false);
    }

private:
    static constexpr float minDb = -48.0f, maxDb = 6.0f;

    static float dbToX(float db, juce::Rectangle<float> r)
    {
        return r.getX() + juce::jlimit(0.0f, 1.0f, (db - minDb) / (maxDb - minDb)) * r.getWidth();
    }

    void drawBar(juce::Graphics& g, juce::Rectangle<float> r, float db, float hold) const
    {
        g.setColour(juce::Colour(0xff0a0c10));
        g.fillRoundedRectangle(r, 2.0f);

        juce::ColourGradient grad(juce::Colour(0xff2bd96b), r.getX(), r.getY(),
                                  juce::Colour(0xffff2a2a), r.getRight(), r.getY(), false);
        grad.addColour(0.667, juce::Colour(0xffe6d62a));   // ab -12 dB
        grad.addColour(0.833, juce::Colour(0xffff8a2a));   // ab  -3 dB
        grad.addColour(0.889, juce::Colour(0xffff2a2a));   // ab   0 dB

        g.saveState();
        g.reduceClipRegion(r.withRight(dbToX(db, r)).toNearestInt());
        g.setGradientFill(grad);
        g.fillRoundedRectangle(r, 2.0f);
        g.restoreState();

        g.setColour(juce::Colours::white.withAlpha(0.9f));
        g.fillRect(dbToX(hold, r) - 1.0f, r.getY(), 2.0f, r.getHeight());
    }

    static void update(float& disp, float& hold, int& counter, float peak)
    {
        const float db = juce::Decibels::gainToDecibels(peak, minDb);
        disp = (db >= disp) ? db : std::max(db, disp - 1.8f);

        if (db >= hold)       { hold = db; counter = 30; }
        else if (counter > 0) { --counter; }
        else                  { hold = std::max(minDb, hold - 0.5f); }
    }

    void timerCallback() override
    {
        update(dispL, holdL, counterL, processor.takePeakL());
        update(dispR, holdR, counterR, processor.takePeakR());
        clip = processor.isClipLatched();
        repaint();
    }

    ModelD3AudioProcessor& processor;
    float dispL = minDb, dispR = minDb, holdL = minDb, holdR = minDb;
    int counterL = 0, counterR = 0;
    bool clip = false;
};

// ==============================================================================
// Editor
// ==============================================================================
class WheelComboBox : public juce::ComboBox
{
public:
    void mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails& wheel) override
    {
        if (wheel.deltaY == 0.0f)
            return;
        const int n = getNumItems();
        if (n <= 0)
            return;
        const int dir = (wheel.deltaY > 0.0f) ? -1 : 1;   // Mausrad nach unten = naechster Eintrag
        setSelectedItemIndex(juce::jlimit(0, n - 1, getSelectedItemIndex() + dir),
                             juce::sendNotificationSync);
    }
};

struct EditorControl
{
    std::unique_ptr<juce::Component> comp;
    std::unique_ptr<juce::Label> label;
    bool isKnob = false;
    bool isBig = false;
};

struct EditorSectionDef
{
    juce::String title;
    int column;
    std::vector<juce::String> ids;
};

struct EditorSection
{
    juce::String title;
    int column = 0;
    juce::Rectangle<int> area;
    std::unique_ptr<juce::Label> header;
    std::vector<EditorControl> controls;
    std::unique_ptr<juce::Component> extra;
    int extraH = 0;
};

class EditorContent : public juce::Component, private juce::AudioProcessorValueTreeState::Listener
{
public:
    static constexpr int kBaseWidth = 960;
    static constexpr int kTopPanelH = 140;

    explicit EditorContent(ModelD3AudioProcessor& p)
        : audioProcessor(p), wtDisplay(p)
    {
        presetBox.addItemList(WavetableBank::getPresetNames(), 1);
        presetAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
            audioProcessor.apvts, "wtPreset", presetBox);
        addAndMakeVisible(presetBox);

        loadBtn.setButtonText("Load .WAV...");
        loadBtn.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff2c5e8a));
        loadBtn.onClick = [this]() { openWavetableChooser(); };
        addAndMakeVisible(loadBtn);

        addAndMakeVisible(wtDisplay);

        static const std::vector<EditorSectionDef> sectionDefs = {
            { "OSC 1", 0, { "osc1On", "osc1Wave", "osc1Oct", "osc1Detune", "osc1Level" } },
            { "OSC 2", 0, { "osc2On", "osc2Wave", "osc2Oct", "osc2Detune", "osc2Level" } },
            { "OSC 3", 0, { "osc3On", "osc3Wave", "osc3Oct", "osc3Detune", "osc3Level" } },
            { "WAVETABLE OSC", 0, { "wtOn", "wtOct", "wtDetune", "wtPos", "wtEnvAmt", "wtLfoWave", "wtLfoRate", "wtLfoAmt", "lfoToFilter", "wtLevel" } },
            { "FILTER", 0, { "cutoff", "resonance", "contour", "filterDrive" } },
            { "FILTER ENV", 0, { "fAttack", "fDecay", "fSustain", "fRelease" } },
            { "AMP ENV", 0, { "aAttack", "aDecay", "aSustain", "aRelease" } },
            { "VOICE", 1, { "polyVoices", "unisonVoices", "unisonDetune", "unisonSpread", "syncMode", "syncPitch", "syncEnv", "analogDrift", "glide", "oversampling" } },
            { "MIXER", 1, { "drive", "masterGain" } },
            { "ARPEGGIATOR", 1, { "arpOn", "arpRate", "arpMode", "arpOctaves", "arpGate", "arpHold" } },
            { "CHORUS", 1, { "chorusRate", "chorusDepth", "chorusMix" } },
            { "DELAY", 1, { "delayTime", "delayFeedback", "delayMix" } },
            { "REVERB", 1, { "revSize", "revDamp", "revMix" } },
            { "NOISE", 1, { "noiseOn", "noiseType", "noiseLevel" } }
        };

        juce::StringArray placed;
        for (const auto& def : sectionDefs)
        {
            EditorSection s;
            s.title = def.title;
            s.column = def.column;
            s.header = std::make_unique<juce::Label>(juce::String(), def.title);
            s.header->setFont(juce::Font(juce::FontOptions(11.0f)));
            s.header->setColour(juce::Label::textColourId, juce::Colour(0xff6fb3e8));
            addAndMakeVisible(*s.header);

            for (const auto& id : def.ids)
            {
                if (auto* param = audioProcessor.apvts.getParameter(id))
                {
                    addControlToSection(s, param);
                    placed.add(id);
                }
            }

            if (def.title == "MIXER")
            {
                s.extra = std::make_unique<LevelMeter>(audioProcessor);
                addAndMakeVisible(*s.extra);
                s.extraH = 40;
            }

            sections.push_back(std::move(s));
        }

        EditorSection other;
        other.column = 1;
        for (auto* param : audioProcessor.getParameters())
        {
            auto* withID = dynamic_cast<juce::AudioProcessorParameterWithID*>(param);
            if (withID == nullptr || withID->paramID == "wtPreset" || placed.contains(withID->paramID))
                continue;

            if (other.header == nullptr)
            {
                other.title = "OTHER";
                other.header = std::make_unique<juce::Label>(juce::String(), "OTHER");
                other.header->setFont(juce::Font(juce::FontOptions(11.0f)));
                other.header->setColour(juce::Label::textColourId, juce::Colour(0xff6fb3e8));
                addAndMakeVisible(*other.header);
            }
            addControlToSection(other, param);
        }
        if (other.header != nullptr)
            sections.push_back(std::move(other));

        setSize(kBaseWidth, layoutSections(kBaseWidth, kTopPanelH + 8) + 8);

        audioProcessor.apvts.addParameterListener("wtOn", this);
        updateWavetablePanel(audioProcessor.apvts.getRawParameterValue("wtOn")->load() > 0.5f);
    }

    ~EditorContent() override
    {
        audioProcessor.apvts.removeParameterListener("wtOn", this);
    }

    std::function<void()> onRelayout;

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff181b20));
        g.setColour(juce::Colour(0xff232730));
        g.fillRect(0, 0, getWidth(), topPanelH);

        for (const auto& s : sections)
        {
            g.setColour(juce::Colour(0xff1f242e));
            g.fillRoundedRectangle(s.area.toFloat(), 6.0f);
            g.setColour(juce::Colour(0xff2c3242));
            g.drawRoundedRectangle(s.area.toFloat(), 6.0f, 1.0f);
        }
    }

    void resized() override
    {
        auto area = getLocalBounds();

        if (topPanelH > 0)
        {
            auto topPanel = area.removeFromTop(topPanelH).reduced(10, 8);

            auto controlRow = topPanel.removeFromTop(30);
            presetBox.setBounds(controlRow.removeFromLeft(controlRow.getWidth() - 125));
            controlRow.removeFromLeft(8);
            loadBtn.setBounds(controlRow);

            topPanel.removeFromTop(8);
            wtDisplay.setBounds(topPanel);
        }

        layoutSections(getWidth(), topPanelH + 8);
    }

private:
    void parameterChanged(const juce::String& parameterID, float newValue) override
    {
        if (parameterID != "wtOn")
            return;

        juce::Component::SafePointer<EditorContent> safeThis(this);
        const bool show = newValue > 0.5f;
        juce::MessageManager::callAsync([safeThis, show]()
        {
            if (safeThis != nullptr)
                safeThis->updateWavetablePanel(show);
        });
    }

    void updateWavetablePanel(bool show)
    {
        topPanelH = show ? kTopPanelH : 0;
        wtDisplay.setVisible(show);
        presetBox.setVisible(show);
        loadBtn.setVisible(show);
        setSize(kBaseWidth, layoutSections(kBaseWidth, topPanelH + 8) + 8);
        if (onRelayout)
            onRelayout();
    }

    void addControlToSection(EditorSection& s, juce::AudioProcessorParameter* param)
    {
        auto* withID = dynamic_cast<juce::AudioProcessorParameterWithID*>(param);

        EditorControl c;
        c.label = std::make_unique<juce::Label>(juce::String(), param->getName(64));
        c.label->setJustificationType(juce::Justification::centred);
        c.label->setColour(juce::Label::textColourId, juce::Colours::lightgrey);
        c.label->setFont(juce::Font(juce::FontOptions(10.0f)));
        addAndMakeVisible(*c.label);

        const auto id = withID != nullptr ? withID->paramID : juce::String();

        if (auto* choice = dynamic_cast<juce::AudioParameterChoice*>(param))
        {
            auto combo = std::make_unique<WheelComboBox>();
            combo->addItemList(choice->choices, 1);
            combo->setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff232730));
            combo->setColour(juce::ComboBox::textColourId, juce::Colours::lightgrey);
            combo->setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff3a4150));
            comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
                audioProcessor.apvts, id, *combo));
            c.comp = std::move(combo);
            c.isKnob = false;
        }
        else if (dynamic_cast<juce::AudioParameterBool*>(param) != nullptr)
        {
            auto btn = std::make_unique<juce::TextButton>();
            btn->setClickingTogglesState(true);
            btn->setColour(juce::TextButton::buttonColourId, juce::Colour(0xff232730));
            btn->setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff2c5e8a));
            btn->setColour(juce::TextButton::textColourOnId, juce::Colours::white);
            buttonAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
                audioProcessor.apvts, id, *btn));
            c.comp = std::move(btn);
            c.isKnob = false;
        }
        else
        {
            auto knob = std::make_unique<juce::Slider>();
            knob->setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
            knob->setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
            knob->setColour(juce::Slider::rotarySliderFillColourId, juce::Colour(0xff2c5e8a));
            knob->setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0xff3a4150));
            knob->setColour(juce::Slider::thumbColourId, juce::Colours::white);
            sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
                audioProcessor.apvts, id, *knob));
            c.comp = std::move(knob);
            c.isKnob = true;
            c.isBig = (id == "cutoff");
        }

        addAndMakeVisible(*c.comp);
        s.controls.push_back(std::move(c));
    }

    int layoutSections(int width, int top)
    {
        const int margin = 12;
        const int gap = 12;
        const int pad4 = 4;
        const int pad2 = 2;
        const int colW = juce::jmax(240, (width - 2 * margin - gap) / 2);
        const int ctrlH = 60;
        const int bigCtrlH = 80;
        const int labelH = 14;
        const int headerH = 20;
        const int sectionGap = 8;
        const int btnH = 26;

        int colY[2] = { top, top };
        const int colX[2] = { margin, margin + colW + gap };

        for (auto& s : sections)
        {
            const int n = static_cast<int>(s.controls.size());
            if (n == 0)
                continue;

            const int col = juce::jlimit(0, 1, s.column);
            const int colsPerRow = juce::jlimit(3, 5, n);

            int maxCtrlH = ctrlH;
            for (const auto& ctl : s.controls)
                if (ctl.isBig)
                    maxCtrlH = bigCtrlH;

            const int rowH = maxCtrlH + labelH + 6;
            const int rows = (n + colsPerRow - 1) / colsPerRow;
            const int h = headerH + rows * rowH + 6 + s.extraH;

            const int x = colX[col];
            int y = colY[col];

            s.area = juce::Rectangle<int>(x - 5, y - 4, colW + 10, h + 4);
            s.header->setBounds(x, y, colW, headerH);
            y += headerH;

            const int cellW = juce::jmax(1, colW / colsPerRow);
            for (int i = 0; i < n; ++i)
            {
                const int c = i % colsPerRow;
                const int r = i / colsPerRow;
                const int cx = x + c * cellW;
                const int cy = y + r * rowH;

                auto& ctl = s.controls[static_cast<size_t>(i)];
                const int ch = ctl.isBig ? bigCtrlH : ctrlH;
                if (ctl.isKnob)
                    ctl.comp->setBounds(cx + pad4, cy + (maxCtrlH - ch) / 2, cellW - 2 * pad4, ch);
                else
                    ctl.comp->setBounds(cx + pad4, cy + (maxCtrlH - btnH) / 2, cellW - 2 * pad4, btnH);
                ctl.label->setBounds(cx + pad4, cy + maxCtrlH + pad2, cellW - 2 * pad4, labelH);
            }

            if (s.extra != nullptr)
                s.extra->setBounds(x + pad4, y + rows * rowH, colW - 2 * pad4, s.extraH - 4);

            colY[col] = y + rows * rowH + s.extraH + sectionGap;
        }

        return juce::jmax(colY[0], colY[1]);
    }

    void openWavetableChooser()
    {
        fileChooser = std::make_unique<juce::FileChooser>(
            "Wavetable WAV-Datei auswaehlen (Serum/Vital 2048-Frame oder Single-Cycle)",
            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
            "*.wav");

        auto flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
        fileChooser->launchAsync(flags, [this](const juce::FileChooser& fc)
        {
            auto file = fc.getResult();
            if (file.existsAsFile())
                audioProcessor.loadWavetableFile(file);
        });
    }

    ModelD3AudioProcessor& audioProcessor;
    Wavetable2DDisplay wtDisplay;
    WheelComboBox presetBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> presetAttachment;
    juce::TextButton loadBtn;
    std::unique_ptr<juce::FileChooser> fileChooser;
    std::vector<EditorSection> sections;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>> buttonAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>> comboAttachments;
    int topPanelH = kTopPanelH;
};

class ModelD3AudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit ModelD3AudioProcessorEditor(ModelD3AudioProcessor& p)
        : AudioProcessorEditor(&p), content(p)
    {
        addAndMakeVisible(content);
        content.onRelayout = [this]()
        {
            resized();
        };

        setResizable(true, true);
        const double ratio = (double) content.getHeight() / (double) EditorContent::kBaseWidth;
        getConstrainer()->setFixedAspectRatio(ratio);
        constexpr int minW = 440;
        constexpr int defaultW = 720;
        constexpr int maxW = 1920;
        setResizeLimits(minW, (int) (minW * ratio), maxW, (int) (maxW * ratio));
        setSize(defaultW, (int) (defaultW * ratio));
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff181b20));
    }

    void resized() override
    {
        const int ch = content.getHeight();
        const float sc = juce::jmin(getWidth() / (float) EditorContent::kBaseWidth,
                                    ch > 0 ? getHeight() / (float) ch : 1.0f);
        const int cw = juce::roundToInt(EditorContent::kBaseWidth * sc);
        const int cht = juce::roundToInt(ch * sc);
        content.setTopLeftPosition((getWidth() - cw) / 2, 0);
        content.setTransform(juce::AffineTransform::scale(sc));
    }

private:
    EditorContent content;
};

juce::AudioProcessorEditor* ModelD3AudioProcessor::createEditor()
{
    return new ModelD3AudioProcessorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ModelD3AudioProcessor();
}
