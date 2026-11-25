#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <memory>
#include <array>

// ==============================================================================
// Analog Thermal & VCO Pitch Drift Generator (Ornstein-Uhlenbeck Random Walk)
// ==============================================================================
class AnalogDriftGenerator
{
public:
    void prepare(double sampleRate, uint32_t seed)
    {
        rng.seed(seed);
        dist = std::normal_distribution<double>(0.0, 1.0);

        slowAlpha = 1.0 - std::exp(-2.0 * juce::MathConstants<double>::pi * 0.16 / sampleRate);
        fastAlpha = 1.0 - std::exp(-2.0 * juce::MathConstants<double>::pi * 2.10 / sampleRate);

        slowState = dist(rng) * 0.35;
        fastState = 0.0;
    }

    double getNextDriftCents(double maxCents)
    {
        if (maxCents <= 0.0001)
            return 0.0;

        double w1 = dist(rng);
        double w2 = dist(rng);

        slowState += slowAlpha * (-slowState) + std::sqrt(2.0 * slowAlpha) * 0.45 * w1;
        fastState += fastAlpha * (-fastState) + std::sqrt(2.0 * fastAlpha) * 0.25 * w2;

        double combined = juce::jlimit(-1.5, 1.5, slowState * 0.80 + fastState * 0.20);
        return combined * maxCents;
    }

private:
    std::mt19937 rng { 42 };
    std::normal_distribution<double> dist { 0.0, 1.0 };
    double slowAlpha = 0.00001;
    double fastAlpha = 0.0001;
    double slowState = 0.0;
    double fastState = 0.0;
};

// ==============================================================================
// Dedicated Wavetable LFO Generator (Smooth & Organic Pad Modulation)
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

    void reset()
    {
        phase = 0.0;
        shCurrent = 0.0f;
        shTarget = 0.0f;
    }

    float processBlock(int numSamples, float rateHz, int waveType)
    {
        if (sr <= 0.0) return 0.0f;
        double inc = (static_cast<double>(rateHz) * static_cast<double>(numSamples)) / sr;
        double prevPhase = phase;
        phase += inc;

        if (phase >= 1.0)
        {
            phase -= std::floor(phase);
            if (waveType == SampleAndHold)
                shTarget = uniDist(rng);
        }

        // Glättung für S&H, damit bei Pads keine harten Klicks entstehen
        shCurrent += (shTarget - shCurrent) * 0.15f;

        const double twoPi = juce::MathConstants<double>::twoPi;
        switch (waveType)
        {
            case Sine:
                return static_cast<float>(std::sin(phase * twoPi));
            case Triangle:
                return static_cast<float>(1.0 - 4.0 * std::abs(std::round(phase - 0.25) - (phase - 0.25)));
            case SawUp:
                return static_cast<float>(2.0 * phase - 1.0);
            case SawDown:
                return static_cast<float>(1.0 - 2.0 * phase);
            case SampleAndHold:
            default:
                juce::ignoreUnused(prevPhase);
                return shCurrent;
        }
    }

private:
    double sr = 44100.0;
    double phase = 0.0;
    float shCurrent = 0.0f;
    float shTarget = 0.0f;
    std::mt19937 rng { 918273 };
    std::uniform_real_distribution<float> uniDist { -1.0f, 1.0f };
};

// ==============================================================================
// High-Precision PolyBLEP + Oversampled Analog Oscillator
// ==============================================================================
class PolyBLEPOscillator
{
public:
    enum Waveform { Saw = 0, Triangle, Square, PulseWide };

    void setSampleRate(double sr) { sampleRate = sr; }
    void reset(double initialPhase = 0.0)
    {
        phase = initialPhase - std::floor(initialPhase);
        lastOutput = 0.0f;
    }

    void setFrequency(double freq)
    {
        frequency = juce::jlimit(10.0, sampleRate * 0.48, freq);
        phaseInc = frequency / sampleRate;
    }

    void setWaveform(int wf) { waveform = wf; }

    float processSample()
    {
        double t = phase;
        double dt = phaseInc;
        float out = 0.0f;

        if (waveform == Saw)
        {
            out = static_cast<float>(2.0 * t - 1.0);
            out -= static_cast<float>(polyBLEP(t, dt));
        }
        else if (waveform == Square || waveform == PulseWide)
        {
            double pw = (waveform == Square) ? 0.5 : 0.25;
            out = (t < pw) ? 1.0f : -1.0f;
            out += static_cast<float>(polyBLEP(t, dt));
            double t2 = std::fmod(t + (1.0 - pw), 1.0);
            out -= static_cast<float>(polyBLEP(t2, dt));
        }
        else
        {
            float sq = (t < 0.5) ? 1.0f : -1.0f;
            sq += static_cast<float>(polyBLEP(t, dt));
            double t2 = std::fmod(t + 0.5, 1.0);
            sq -= static_cast<float>(polyBLEP(t2, dt));

            lastOutput = static_cast<float>(4.0 * dt * sq + (1.0 - 4.0 * dt) * lastOutput);
            out = juce::jlimit(-1.0f, 1.0f, lastOutput * 1.15f);
        }

        phase += phaseInc;
        if (phase >= 1.0)
            phase -= 1.0;

        return out;
    }

private:
    static double polyBLEP(double t, double dt)
    {
        if (dt <= 0.0) return 0.0;
        if (t < dt)
        {
            t /= dt;
            return t + t - t * t - 1.0;
        }
        else if (t > 1.0 - dt)
        {
            t = (t - 1.0) / dt;
            return t * t + t + t + 1.0;
        }
        return 0.0;
    }

    double sampleRate = 44100.0;
    double frequency = 440.0;
    double phase = 0.0;
    double phaseInc = 0.01;
    float lastOutput = 0.0f;
    int waveform = Saw;
};

// ==============================================================================
// Wavetable Data & 50 Pad-Focused Built-In Preset Wavetables + WAV Loader
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
            "01: Solina String Ensemble",
            "02: CS-80 Velvet Brass Swell",
            "03: OB-Xa Warm Analog Pad",
            "04: Juno-106 Chorus Silk",
            "05: Prophet-5 Warm Saw-Tri",
            "06: Jupiter-8 Lush Strings",
            "07: MiniMoog Soft Ladder Bloom",
            "08: PWM Creamy Ribbon Pad",
            "09: Dual-Detuned Analog Drift",
            "10: Warm Sub-Octave Pillow",
            "11: Mellotron Tape Choir",
            "12: Cathedral Angelic Choir",
            "13: Vowel Formant A-O-U Morph",
            "14: Breath & Air Silk Pad",
            "15: Whispering Formant Cloud",
            "16: Soprano & Alto Blend",
            "17: Ethereal Glass Harmonics",
            "18: PPG Wave 2.2 Crystal Pad",
            "19: Frozen Ice Bell Shimmer",
            "20: Celestial Harp Harmonics",
            "21: Shimmering Octave Mist",
            "22: Warm Rhodes Tine Pad",
            "23: DX7 Soft Sine Bloom",
            "24: FM Velvet Electric Pad",
            "25: FM Warm Bell Atmosphere",
            "26: Bowed Cello Section",
            "27: Chamber Strings Adagio",
            "28: Warm French Horn Swell",
            "29: Bass Clarinet & Woodwind",
            "30: Bowed Glass & Waterphone",
            "31: Hammond Warm Drawbars",
            "32: Pipe Organ Flutes 8'+4'",
            "33: Soft Reed Harmonium",
            "34: Octave Fifths Cathedral",
            "35: Warm Tape Saturation Morph",
            "36: Even-Harmonic Tube Glow",
            "37: Chebyshev Silk Overtones",
            "38: West-Coast Soft Wavefold",
            "39: Phase-Shift Warm Cloud",
            "40: Spectral Comb Breeze",
            "41: Golden Ratio Harmonics",
            "42: Fibonacci Overtones Pad",
            "43: Sub-Fifths Ambient Drone",
            "44: Deep Space Nebula Morph",
            "45: Analog Horizon Sunrise",
            "46: Arctic Aurora Shimmer",
            "47: Velvet Night Nocturne",
            "48: Liquid Silk Chorus Wave",
            "49: Dreamy Lo-Fi Tape Pad",
            "50: Cosmic Evolving Symphony",
            "51: [Custom Loaded WAV]"
        };
    }

    static std::shared_ptr<WavetableData> createBuiltInPreset(int index)
    {
        auto data = std::make_shared<WavetableData>();
        const int numFrames = 64;
        const int N = WavetableData::tableSize;
        data->frames.resize(numFrames, std::vector<float>(N, 0.0f));
        auto names = getPresetNames();
        data->name = names[juce::jlimit(0, numBuiltInPresets - 1, index)];

        const double twoPi = juce::MathConstants<double>::twoPi;

        // Bandlimitierte harmonische Synthese verhindert digitales Aliasing und sorgt für samtige Pad-Textur
        for (int f = 0; f < numFrames; ++f)
        {
            double m = static_cast<double>(f) / static_cast<double>(numFrames - 1);
            float peak = 0.0001f;

            for (int i = 0; i < N; ++i)
            {
                double ph = static_cast<double>(i) / static_cast<double>(N);
                double w = ph * twoPi;
                double val = 0.0;

                switch (index)
                {
                    case 0: // 01: Solina String Ensemble
                    {
                        for (int h = 1; h <= 22; ++h)
                        {
                            double phaseMod = std::sin(m * twoPi + h * 0.85) * 1.4;
                            double env = std::exp(-h * (0.09 - 0.04 * m));
                            val += (env / std::pow(h, 0.7)) * std::sin(h * w + phaseMod);
                        }
                        break;
                    }
                    case 1: // 02: CS-80 Velvet Brass Swell
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
                    case 2: // 03: OB-Xa Warm Analog Pad
                    {
                        for (int h = 1; h <= 20; ++h)
                        {
                            double spread = m * h * 0.45;
                            val += (1.0 / std::pow(h, 0.85)) * (std::sin(h * w) + 0.65 * std::sin(h * w + spread));
                        }
                        break;
                    }
                    case 3: // 04: Juno-106 Chorus Silk
                    {
                        for (int h = 1; h <= 18; ++h)
                        {
                            double c1 = std::sin(h * w);
                            double c2 = std::sin(h * w + m * 2.1 * h);
                            double c3 = std::sin(h * w - m * 1.7 * h);
                            val += (1.0 / std::pow(h, 0.8)) * (c1 + 0.55 * c2 + 0.55 * c3);
                        }
                        break;
                    }
                    case 4: // 05: Prophet-5 Warm Saw-Tri
                    {
                        for (int h = 1; h <= 24; ++h)
                        {
                            double triAmp = (h % 2 == 1) ? (1.0 / (h * h)) * ((h % 4 == 1) ? 1.0 : -1.0) : 0.0;
                            double sawAmp = (1.0 / h) * std::exp(-h * 0.05);
                            val += ((1.0 - m) * triAmp * 1.4 + m * sawAmp) * std::sin(h * w);
                        }
                        break;
                    }
                    case 5: // 06: Jupiter-8 Lush Strings
                    {
                        for (int h = 1; h <= 24; ++h)
                        {
                            double detunePh = std::sin(m * 3.14159 * 2.0 + h * 0.5) * 1.1;
                            double rolloff = std::exp(-h * (0.08 - 0.03 * m));
                            val += (rolloff / std::pow(h, 0.75)) * (std::sin(h * w) + std::sin(h * w + detunePh));
                        }
                        break;
                    }
                    case 6: // 07: MiniMoog Soft Ladder Bloom
                    {
                        double fc = 2.0 + m * 18.0;
                        for (int h = 1; h <= 24; ++h)
                        {
                            double ladder = 1.0 / std::pow(1.0 + std::pow(h / fc, 2.0), 2.0);
                            val += (ladder / std::pow(h, 0.65)) * std::sin(h * w);
                        }
                        break;
                    }
                    case 7: // 08: PWM Creamy Ribbon Pad
                    {
                        double pw = 0.5 - 0.36 * m;
                        for (int h = 1; h <= 24; ++h)
                        {
                            double pwmAmp = (2.0 / (h * 3.14159)) * std::sin(h * 3.14159 * pw) * std::exp(-h * 0.045);
                            val += pwmAmp * std::cos(h * w);
                        }
                        break;
                    }
                    case 8: // 09: Dual-Detuned Analog Drift
                    {
                        for (int h = 1; h <= 20; ++h)
                        {
                            double d = m * 3.14159 * (0.3 + 0.15 * h);
                            val += (1.0 / std::pow(h, 0.8)) * (std::sin(h * w) + std::sin(h * w + d));
                        }
                        break;
                    }
                    case 9: // 10: Warm Sub-Octave Pillow
                    {
                        val = std::sin(w)
                            + (0.45 + 0.25 * m) * std::sin(2.0 * w)
                            + (0.20 * m) * std::sin(3.0 * w)
                            + (0.15 * m * m) * std::sin(4.0 * w);
                        break;
                    }
                    case 10: // 11: Mellotron Tape Choir
                    {
                        double f1 = 3.0 + m * 2.5;
                        double f2 = 7.0 + m * 4.0;
                        for (int h = 1; h <= 16; ++h)
                        {
                            double g1 = std::exp(-std::pow((h - f1) / 1.8, 2.0));
                            double g2 = 0.65 * std::exp(-std::pow((h - f2) / 2.2, 2.0));
                            val += (0.28 / h + g1 + g2) * std::sin(h * w + 0.15 * std::sin(h * 1.7));
                        }
                        break;
                    }
                    case 11: // 12: Cathedral Angelic Choir
                    {
                        double f1 = 4.0 + std::sin(m * 3.14159) * 3.0;
                        double f2 = 10.0 + m * 5.0;
                        for (int h = 1; h <= 20; ++h)
                        {
                            double g1 = std::exp(-std::pow((h - f1) / 1.6, 2.0));
                            double g2 = 0.7 * std::exp(-std::pow((h - f2) / 2.4, 2.0));
                            val += (0.2 / h + g1 + g2) * std::sin(h * w);
                        }
                        break;
                    }
                    case 12: // 13: Vowel Formant A-O-U Morph
                    {
                        double f1 = 6.0 - m * 3.5;
                        double f2 = 11.0 - m * 4.5;
                        for (int h = 1; h <= 18; ++h)
                        {
                            double formant = std::exp(-std::pow((h - f1) / 1.5, 2.0))
                                           + 0.7 * std::exp(-std::pow((h - f2) / 2.0, 2.0));
                            val += (0.25 / h + formant) * std::sin(h * w);
                        }
                        break;
                    }
                    case 13: // 14: Breath & Air Silk Pad
                    {
                        for (int h = 1; h <= 24; ++h)
                        {
                            double air = (h > 10) ? (m * 0.28 * std::sin(h * 2.3 + m * 4.0)) : 0.0;
                            double body = std::exp(-h * 0.16);
                            val += (body + air) * std::sin(h * w + h * 0.3);
                        }
                        break;
                    }
                    case 14: // 15: Whispering Formant Cloud
                    {
                        for (int h = 1; h <= 20; ++h)
                        {
                            double peak1 = std::exp(-std::pow((h - (5.0 + m * 7.0)) / 2.3, 2.0));
                            val += (0.3 / h + 0.85 * peak1) * std::sin(h * w + m * h * 0.6);
                        }
                        break;
                    }
                    case 15: // 16: Soprano & Alto Blend
                    {
                        for (int h = 1; h <= 18; ++h)
                        {
                            double alto = std::exp(-std::pow((h - 4.0) / 1.5, 2.0)) * (1.0 - 0.5 * m);
                            double sop  = std::exp(-std::pow((h - 8.0) / 2.0, 2.0)) * (0.3 + 0.7 * m);
                            val += (0.25 / h + alto + sop) * std::sin(h * w);
                        }
                        break;
                    }
                    case 16: // 17: Ethereal Glass Harmonics
                    {
                        val = std::sin(w)
                            + (0.3 + 0.4 * m) * std::sin(3.0 * w)
                            + (0.15 + 0.45 * m) * std::sin(6.0 * w)
                            + (0.35 * m * m) * std::sin(9.0 * w);
                        break;
                    }
                    case 17: // 18: PPG Wave 2.2 Crystal Pad
                    {
                        for (int h = 1; h <= 16; ++h)
                        {
                            double crystal = std::sin(m * 4.5 + h * 0.9);
                            double weight = (crystal > 0.0 ? crystal : 0.0) / std::sqrt(static_cast<double>(h));
                            val += ((1.0 / (h * h)) + 0.55 * weight) * std::sin(h * w);
                        }
                        break;
                    }
                    case 18: // 19: Frozen Ice Bell Shimmer
                    {
                        val = std::sin(w)
                            + 0.45 * std::sin(2.0 * w)
                            + (0.4 * m) * std::sin(5.0 * w + m * 1.5)
                            + (0.35 * m) * std::sin(7.0 * w)
                            + (0.25 * m * m) * std::sin(11.0 * w);
                        break;
                    }
                    case 19: // 20: Celestial Harp Harmonics
                    {
                        for (int h = 1; h <= 16; ++h)
                        {
                            double damp = std::exp(-h * (0.24 - 0.16 * m));
                            val += damp * std::sin(h * w);
                        }
                        break;
                    }
                    case 20: // 21: Shimmering Octave Mist
                    {
                        val = std::sin(w)
                            + 0.55 * std::sin(2.0 * w)
                            + (0.35 + 0.3 * m) * std::sin(4.0 * w)
                            + (0.45 * m) * std::sin(8.0 * w)
                            + (0.30 * m * m) * std::sin(12.0 * w);
                        break;
                    }
                    case 21: // 22: Warm Rhodes Tine Pad
                    {
                        val = std::sin(w)
                            + 0.42 * std::sin(2.0 * w)
                            + (0.22 * (1.0 - 0.4 * m)) * std::sin(3.0 * w)
                            + (0.28 * m) * std::sin(7.0 * w);
                        break;
                    }
                    case 22: // 23: DX7 Soft Sine Bloom
                    {
                        double idx = 0.15 + m * 1.85;
                        val = std::sin(w + idx * std::sin(2.0 * w)) + 0.25 * std::sin(w);
                        break;
                    }
                    case 23: // 24: FM Velvet Electric Pad
                    {
                        double idx = 0.25 + m * 2.1;
                        val = std::sin(w + idx * std::sin(w) * std::cos(2.0 * w));
                        break;
                    }
                    case 24: // 25: FM Warm Bell Atmosphere
                    {
                        double idx = 0.2 + m * 1.9;
                        val = 0.75 * std::sin(w + idx * std::sin(3.0 * w)) + 0.35 * std::sin(2.0 * w);
                        break;
                    }
                    case 25: // 26: Bowed Cello Section
                    {
                        for (int h = 1; h <= 20; ++h)
                        {
                            double bow = std::exp(-h * 0.08) * (1.0 / std::pow(h, 0.75));
                            val += bow * std::sin(h * w + m * h * 0.55);
                        }
                        break;
                    }
                    case 26: // 27: Chamber Strings Adagio
                    {
                        for (int h = 1; h <= 18; ++h)
                        {
                            double vibr = std::sin(m * twoPi + h * 0.6) * 0.8;
                            val += (std::exp(-h * 0.11) / std::sqrt(static_cast<double>(h))) * std::sin(h * w + vibr);
                        }
                        break;
                    }
                    case 27: // 28: Warm French Horn Swell
                    {
                        double cutoff = 2.5 + m * 11.0;
                        for (int h = 1; h <= 16; ++h)
                        {
                            double att = 1.0 / (1.0 + std::pow(h / cutoff, 2.5));
                            val += (att / std::pow(h, 0.65)) * std::sin(h * w);
                        }
                        break;
                    }
                    case 28: // 29: Bass Clarinet & Woodwind
                    {
                        for (int h = 1; h <= 15; h += 2)
                        {
                            double weight = std::exp(-h * (0.14 - 0.07 * m));
                            val += (weight / std::pow(h, 0.7)) * std::sin(h * w);
                        }
                        val += 0.22 * m * std::sin(2.0 * w);
                        break;
                    }
                    case 29: // 30: Bowed Glass & Waterphone
                    {
                        val = std::sin(w)
                            + (0.5 * m) * std::sin(3.0 * w + m * 2.0)
                            + (0.35 * m) * std::sin(5.0 * w - m * 1.5)
                            + (0.25 * m * m) * std::sin(8.0 * w);
                        break;
                    }
                    case 30: // 31: Hammond Warm Drawbars
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
                    case 31: // 32: Pipe Organ Flutes 8'+4'
                    {
                        val = std::sin(w)
                            + 0.65 * std::sin(2.0 * w)
                            + (0.35 * m) * std::sin(3.0 * w)
                            + (0.45 * m) * std::sin(4.0 * w)
                            + (0.20 * m * m) * std::sin(6.0 * w);
                        break;
                    }
                    case 32: // 33: Soft Reed Harmonium
                    {
                        for (int h = 1; h <= 16; ++h)
                        {
                            double roll = std::exp(-h * (0.15 - 0.08 * m));
                            val += (roll / std::sqrt(static_cast<double>(h))) * std::sin(h * w);
                        }
                        break;
                    }
                    case 33: // 34: Octave Fifths Cathedral
                    {
                        val = std::sin(w)
                            + 0.60 * std::sin(2.0 * w)
                            + (0.45 + 0.25 * m) * std::sin(3.0 * w)
                            + (0.35 * m) * std::sin(4.0 * w)
                            + (0.30 * m) * std::sin(6.0 * w);
                        break;
                    }
                    case 34: // 35: Warm Tape Saturation Morph
                    {
                        double raw = std::sin(w) + 0.45 * std::sin(2.0 * w) + 0.25 * std::sin(3.0 * w);
                        double drive = 1.0 + m * 3.2;
                        val = std::tanh(raw * drive) / std::tanh(drive);
                        break;
                    }
                    case 35: // 36: Even-Harmonic Tube Glow
                    {
                        val = std::sin(w)
                            + (0.2 + 0.45 * m) * std::sin(2.0 * w)
                            + (0.08 + 0.28 * m) * std::sin(4.0 * w)
                            + (0.15 * m * m) * std::sin(6.0 * w);
                        break;
                    }
                    case 36: // 37: Chebyshev Silk Overtones
                    {
                        double x = std::sin(w);
                        double t2 = 2.0 * x * x - 1.0;
                        double t3 = 4.0 * x * x * x - 3.0 * x;
                        val = x + (0.4 * m) * t2 + (0.25 * m * m) * t3;
                        break;
                    }
                    case 37: // 38: West-Coast Soft Wavefold
                    {
                        double gain = 1.0 + m * 2.4;
                        val = std::sin(gain * std::sin(w));
                        break;
                    }
                    case 38: // 39: Phase-Shift Warm Cloud
                    {
                        for (int h = 1; h <= 14; ++h)
                        {
                            double shift = std::sin(m * twoPi + h * 0.4) * 2.0;
                            val += (std::exp(-h * 0.12) / std::sqrt(static_cast<double>(h))) * std::sin(h * w + shift);
                        }
                        break;
                    }
                    case 39: // 40: Spectral Comb Breeze
                    {
                        double notch = 2.0 + m * 10.0;
                        for (int h = 1; h <= 18; ++h)
                        {
                            double comb = 0.5 + 0.5 * std::cos((h / notch) * 3.14159);
                            val += (comb * std::exp(-h * 0.09) / std::sqrt(static_cast<double>(h))) * std::sin(h * w);
                        }
                        break;
                    }
                    case 40: // 41: Golden Ratio Harmonics
                    {
                        val = std::sin(w)
                            + 0.45 * std::sin(2.0 * w)
                            + (0.4 * m) * std::sin(3.0 * w)
                            + (0.35 * m) * std::sin(5.0 * w)
                            + (0.25 * m * m) * std::sin(8.0 * w);
                        break;
                    }
                    case 41: // 42: Fibonacci Overtones Pad
                    {
                        const int fib[7] = { 1, 2, 3, 5, 8, 13, 21 };
                        for (int k = 0; k < 7; ++k)
                        {
                            double wgt = std::exp(-k * (0.45 - 0.25 * m));
                            val += wgt * std::sin(fib[k] * w);
                        }
                        break;
                    }
                    case 42: // 43: Sub-Fifths Ambient Drone
                    {
                        val = std::sin(w)
                            + 0.55 * std::sin(2.0 * w)
                            + (0.35 + 0.25 * m) * std::sin(3.0 * w)
                            + (0.25 * m) * std::sin(6.0 * w);
                        break;
                    }
                    case 43: // 44: Deep Space Nebula Morph
                    {
                        for (int h = 1; h <= 16; ++h)
                        {
                            double swell = 0.5 + 0.5 * std::sin(m * 4.0 + h * 0.75);
                            val += (swell * std::exp(-h * 0.11) / std::sqrt(static_cast<double>(h))) * std::sin(h * w);
                        }
                        break;
                    }
                    case 44: // 45: Analog Horizon Sunrise
                    {
                        for (int h = 1; h <= 20; ++h)
                        {
                            double horizon = std::exp(-h * (0.28 - 0.22 * m));
                            val += (horizon / std::pow(h, 0.7)) * std::sin(h * w);
                        }
                        break;
                    }
                    case 45: // 46: Arctic Aurora Shimmer
                    {
                        for (int h = 1; h <= 18; ++h)
                        {
                            double band = std::exp(-std::pow((h - (3.0 + m * 10.0)) / 3.0, 2.0));
                            val += (0.35 / h + 0.65 * band / std::sqrt(static_cast<double>(h))) * std::sin(h * w + h * 0.4);
                        }
                        break;
                    }
                    case 46: // 47: Velvet Night Nocturne
                    {
                        val = std::sin(w)
                            + (0.4 + 0.2 * m) * std::sin(2.0 * w)
                            + (0.25 * m) * std::sin(3.0 * w)
                            + (0.12 * m) * std::sin(4.0 * w);
                        break;
                    }
                    case 47: // 48: Liquid Silk Chorus Wave
                    {
                        for (int h = 1; h <= 14; ++h)
                        {
                            double ph1 = std::sin(m * twoPi + h * 0.5);
                            double ph2 = std::cos(m * twoPi - h * 0.4);
                            val += (std::exp(-h * 0.13) / h) * (std::sin(h * w + ph1) + std::sin(h * w + ph2));
                        }
                        break;
                    }
                    case 48: // 49: Dreamy Lo-Fi Tape Pad
                    {
                        for (int h = 1; h <= 12; ++h)
                        {
                            double wobble = std::sin(h * 1.9 + m * 3.0) * 0.35;
                            val += (std::exp(-h * 0.18) / std::pow(h, 0.8)) * std::sin(h * w + wobble);
                        }
                        break;
                    }
                    default: // 50: Cosmic Evolving Symphony
                    {
                        for (int h = 1; h <= 20; ++h)
                        {
                            double wave1 = 0.5 + 0.5 * std::sin(m * twoPi + h * 0.45);
                            double wave2 = 0.5 + 0.5 * std::cos(m * 3.14159 + h * 0.3);
                            val += (wave1 * wave2 * std::exp(-h * 0.08) / std::sqrt(static_cast<double>(h))) * std::sin(h * w);
                        }
                        break;
                    }
                }

                float fVal = static_cast<float>(val);
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

        std::vector<float> mono(totalSamples, 0.0f);
        for (int ch = 0; ch < tempBuffer.getNumChannels(); ++ch)
        {
            const float* src = tempBuffer.getReadPointer(ch);
            for (int i = 0; i < totalSamples; ++i)
                mono[i] += src[i];
        }
        if (tempBuffer.getNumChannels() > 1)
        {
            float scale = 1.0f / static_cast<float>(tempBuffer.getNumChannels());
            for (float& v : mono) v *= scale;
        }

        auto data = std::make_shared<WavetableData>();
        data->name = "WAV: " + file.getFileNameWithoutExtension();

        if (totalSamples >= WavetableData::tableSize && (totalSamples % WavetableData::tableSize == 0))
        {
            int numFrames = std::min(256, totalSamples / WavetableData::tableSize);
            data->frames.resize(numFrames, std::vector<float>(WavetableData::tableSize, 0.0f));

            for (int f = 0; f < numFrames; ++f)
            {
                int offset = f * WavetableData::tableSize;
                float peak = 0.0001f;
                for (int i = 0; i < WavetableData::tableSize; ++i)
                {
                    float val = mono[offset + i];
                    data->frames[f][i] = val;
                    peak = std::max(peak, std::abs(val));
                }
                if (peak > 0.001f)
                {
                    for (int i = 0; i < WavetableData::tableSize; ++i)
                        data->frames[f][i] /= peak;
                }
            }
        }
        else
        {
            int numFrames = (totalSamples <= 4096) ? 1 : std::min(64, std::max(1, totalSamples / 512));
            int chunkLen = std::max(16, totalSamples / numFrames);
            data->frames.resize(numFrames, std::vector<float>(WavetableData::tableSize, 0.0f));

            for (int f = 0; f < numFrames; ++f)
            {
                int start = f * chunkLen;
                int len = std::min(chunkLen, totalSamples - start);
                if (len < 4) break;

                float peak = 0.0001f;
                for (int i = 0; i < WavetableData::tableSize; ++i)
                {
                    double pos = (static_cast<double>(i) / static_cast<double>(WavetableData::tableSize)) * static_cast<double>(len - 1);
                    int idx0 = static_cast<int>(pos);
                    int idx1 = std::min(len - 1, idx0 + 1);
                    float frac = static_cast<float>(pos - idx0);

                    float val = mono[start + idx0] * (1.0f - frac) + mono[start + idx1] * frac;
                    data->frames[f][i] = val;
                    peak = std::max(peak, std::abs(val));
                }

                if (peak > 0.001f)
                {
                    for (int i = 0; i < WavetableData::tableSize; ++i)
                        data->frames[f][i] /= peak;
                }
            }
        }

        if (data->frames.empty())
            return nullptr;

        return data;
    }
};

// ==============================================================================
// High-Resolution Wavetable Oscillator (2D Frame Morphing + Cubic Hermite)
// ==============================================================================
class WavetableOscillator
{
public:
    void setSampleRate(double sr) { sampleRate = sr; }
    void reset(double initialPhase = 0.0) { phase = initialPhase - std::floor(initialPhase); }

    void setFrequency(double freq)
    {
        frequency = juce::jlimit(10.0, sampleRate * 0.48, freq);
        phaseInc = frequency / sampleRate;
    }

    void setTable(std::shared_ptr<WavetableData> newTable)
    {
        if (newTable != nullptr && !newTable->frames.empty())
            std::atomic_store(&currentTable, newTable);
    }

    std::shared_ptr<WavetableData> getTable() const
    {
        return std::atomic_load(&currentTable);
    }

    float processSample(float morphPos)
    {
        auto table = std::atomic_load(&currentTable);
        if (table == nullptr || table->frames.empty())
            return 0.0f;

        const int numFrames = static_cast<int>(table->frames.size());
        const float clampedMorph = juce::jlimit(0.0f, 1.0f, morphPos);

        float frameFloat = clampedMorph * static_cast<float>(numFrames - 1);
        int frameA = static_cast<int>(frameFloat);
        int frameB = std::min(numFrames - 1, frameA + 1);
        float frameFrac = frameFloat - static_cast<float>(frameA);

        double tablePos = phase * static_cast<double>(WavetableData::tableSize);
        int idx0 = static_cast<int>(tablePos) & (WavetableData::tableSize - 1);
        float frac = static_cast<float>(tablePos - std::floor(tablePos));

        float sampleA = readHermite(table->frames[frameA], idx0, frac);
        float sampleB = (frameA == frameB) ? sampleA : readHermite(table->frames[frameB], idx0, frac);

        phase += phaseInc;
        if (phase >= 1.0)
            phase -= 1.0;

        return sampleA * (1.0f - frameFrac) + sampleB * frameFrac;
    }

private:
    static float readHermite(const std::vector<float>& frame, int idx, float frac)
    {
        constexpr int mask = WavetableData::tableSize - 1;
        float ym1 = frame[(idx - 1) & mask];
        float y0  = frame[idx & mask];
        float y1  = frame[(idx + 1) & mask];
        float y2  = frame[(idx + 2) & mask];

        float c0 = y0;
        float c1 = 0.5f * (y1 - ym1);
        float c2 = ym1 - 2.5f * y0 + 2.0f * y1 - 0.5f * y2;
        float c3 = 0.5f * (y2 - ym1) + 1.5f * (y0 - y1);

        return ((c3 * frac + c2) * frac + c1) * frac + c0;
    }

    std::shared_ptr<WavetableData> currentTable;
    double sampleRate = 44100.0;
    double frequency = 440.0;
    double phase = 0.0;
    double phaseInc = 0.01;
};

// ==============================================================================
// Unison Sub-Voice (3 Analog PolyBLEP Oscs + 1 Wavetable Osc + Drift per Sub-Voice)
// ==============================================================================
struct UnisonSubVoice
{
    static constexpr int maxUnison = 7;

    PolyBLEPOscillator oscs[3];
    AnalogDriftGenerator driftGens[3];
    WavetableOscillator wtOsc;
    AnalogDriftGenerator wtDriftGen;

    void prepare(double sampleRate, int polyVoiceIdx, int unisonIdx)
    {
        uint32_t baseSeed = static_cast<uint32_t>(104729 + polyVoiceIdx * 15401 + unisonIdx * 3571);
        double initPhaseOffset = std::fmod((polyVoiceIdx * 0.2360679 + unisonIdx * 0.38196601), 1.0);

        for (int o = 0; o < 3; ++o)
        {
            oscs[o].setSampleRate(sampleRate);
            oscs[o].reset(std::fmod(initPhaseOffset + o * 0.27, 1.0));
            driftGens[o].prepare(sampleRate, baseSeed + static_cast<uint32_t>(o * 1301));
        }

        wtOsc.setSampleRate(sampleRate);
        wtOsc.reset(std::fmod(initPhaseOffset + 0.13, 1.0));
        wtDriftGen.prepare(sampleRate, baseSeed + 4999u);
    }
};

// ==============================================================================
// Polyphonic Synth Voice (8-Voice Polyphony, each with up to 7 Unison Sub-Voices)
// ==============================================================================
struct PolySynthVoice
{
    static constexpr int maxPolyVoices = 8;

    bool active = false;
    int midiNote = -1;
    float velocity = 1.0f;
    double currentFreq = 110.0;
    double targetFreq = 110.0;
    uint64_t noteOnAge = 0;

    std::array<UnisonSubVoice, UnisonSubVoice::maxUnison> subVoices;
    juce::ADSR ampEnv;
    juce::ADSR filterEnv;

    juce::dsp::LadderFilter<float> ladderFilterL;
    juce::dsp::LadderFilter<float> ladderFilterR;

    void prepare(double sampleRate, int maxBlockSize, int polyIndex)
    {
        for (int u = 0; u < UnisonSubVoice::maxUnison; ++u)
            subVoices[u].prepare(sampleRate, polyIndex, u);

        ampEnv.setSampleRate(sampleRate);
        filterEnv.setSampleRate(sampleRate);
        ampEnv.reset();
        filterEnv.reset();

        juce::dsp::ProcessSpec monoSpec { sampleRate, static_cast<juce::uint32>(maxBlockSize), 1 };
        ladderFilterL.prepare(monoSpec);
        ladderFilterR.prepare(monoSpec);
        ladderFilterL.setMode(juce::dsp::LadderFilterMode::LPF24);
        ladderFilterR.setMode(juce::dsp::LadderFilterMode::LPF24);
        ladderFilterL.setDrive(1.2f);
        ladderFilterR.setDrive(1.2f);
        ladderFilterL.reset();
        ladderFilterR.reset();
    }

    void startNote(int note, float vel, double lastGlobalFreq, bool legatoGlide, uint64_t age)
    {
        midiNote = note;
        velocity = vel;
        noteOnAge = age;
        targetFreq = juce::MidiMessage::getMidiNoteInHertz(note);

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
// Model D-3 AudioProcessor (v2.1.0 - Pad Edition + WT LFO + 50 Pad Wavetables)
// ==============================================================================
class ModelD3AudioProcessor : public juce::AudioProcessor
{
public:
    ModelD3AudioProcessor()
        : AudioProcessor(BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true)),
          apvts(*this, nullptr, "PARAMETERS", createParameterLayout())
    {
        for (int i = 0; i < WavetableBank::numBuiltInPresets; ++i)
            builtInTables[i] = WavetableBank::createBuiltInPreset(i);

        activeWavetable = builtInTables[0];
        applyWavetableToAllVoices(activeWavetable);
    }

    ~ModelD3AudioProcessor() override = default;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

        juce::StringArray waveChoices { "Sawtooth", "Triangle", "Square", "Pulse 25%" };
        juce::StringArray octaveChoices { "32'", "16'", "8'", "4'", "2'" };
        juce::StringArray lfoWaveChoices { "Sine", "Triangle", "Saw Up", "Saw Down", "Smooth S&H" };
        juce::StringArray arpRateChoices { "1/4", "1/8", "1/8T", "1/16", "1/32" };
        juce::StringArray arpModeChoices { "Up", "Down", "Up/Down" };

        // Polyphony & Unison
        params.push_back(std::make_unique<juce::AudioParameterInt>("polyVoices", "Polyphony Voices", 1, 8, 8));
        params.push_back(std::make_unique<juce::AudioParameterInt>("unisonVoices", "Unison Voices", 1, 7, 4));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("unisonDetune", "Unison Detune (ct)", 0.0f, 60.0f, 14.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("unisonSpread", "Unison Stereo Spread", 0.0f, 1.0f, 0.85f));

        // Oscillator 1
        params.push_back(std::make_unique<juce::AudioParameterBool>("osc1On", "Osc 1 On", true));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc1Wave", "Osc 1 Wave", waveChoices, 0));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc1Oct", "Osc 1 Octave", octaveChoices, 2));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc1Detune", "Osc 1 Detune (ct)", -50.0f, 50.0f, -4.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc1Level", "Osc 1 Level", 0.0f, 1.0f, 0.65f));

        // Oscillator 2
        params.push_back(std::make_unique<juce::AudioParameterBool>("osc2On", "Osc 2 On", true));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc2Wave", "Osc 2 Wave", waveChoices, 1));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc2Oct", "Osc 2 Octave", octaveChoices, 2));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc2Detune", "Osc 2 Detune (ct)", -50.0f, 50.0f, 4.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc2Level", "Osc 2 Level", 0.0f, 1.0f, 0.60f));

        // Oscillator 3
        params.push_back(std::make_unique<juce::AudioParameterBool>("osc3On", "Osc 3 On", true));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc3Wave", "Osc 3 Wave", waveChoices, 0));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("osc3Oct", "Osc 3 Octave", octaveChoices, 1));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc3Detune", "Osc 3 Detune (ct)", -50.0f, 50.0f, 0.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("osc3Level", "Osc 3 Level", 0.0f, 1.0f, 0.50f));

        // Oscillator 4: Wavetable Oscillator (50 Pad Wavetables + WAV Loader + Dedicated LFO)
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
        params.push_back(std::make_unique<juce::AudioParameterFloat>("wtLevel", "WT Osc Level", 0.0f, 1.0f, 0.85f));

        // Analog Drift, Glide & Mixer Drive
        params.push_back(std::make_unique<juce::AudioParameterFloat>("analogDrift", "Analog Drift (ct)", 0.0f, 25.0f, 5.5f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("glide", "Glide (ms)", 0.0f, 500.0f, 25.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("drive", "Mixer Drive", 1.0f, 6.0f, 1.6f));

        // 24dB Ladder Filter & Filter Envelope
        params.push_back(std::make_unique<juce::AudioParameterFloat>("cutoff", "Filter Cutoff (Hz)",
            juce::NormalisableRange<float>(20.0f, 18000.0f, 0.1f, 0.3f), 1450.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("resonance", "Filter Emphasis", 0.0f, 0.98f, 0.22f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("contour", "Filter Contour", 0.0f, 1.0f, 0.40f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("fAttack", "Filter Attack (s)",
            juce::NormalisableRange<float>(0.001f, 4.0f, 0.0001f, 0.4f), 0.65f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("fDecay", "Filter Decay (s)",
            juce::NormalisableRange<float>(0.005f, 5.0f, 0.0001f, 0.4f), 1.40f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("fSustain", "Filter Sustain", 0.0f, 1.0f, 0.55f));

        // Loudness (VCA) Envelope (Pad-friendly default swell)
        params.push_back(std::make_unique<juce::AudioParameterFloat>("aAttack", "Amp Attack (s)",
            juce::NormalisableRange<float>(0.001f, 4.0f, 0.0001f, 0.4f), 0.45f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("aDecay", "Amp Decay (s)",
            juce::NormalisableRange<float>(0.005f, 5.0f, 0.0001f, 0.4f), 1.20f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("aSustain", "Amp Sustain", 0.0f, 1.0f, 0.85f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("aRelease", "Amp Release (s)",
            juce::NormalisableRange<float>(0.003f, 6.0f, 0.0001f, 0.4f), 1.35f));

        // Arpeggiator
        params.push_back(std::make_unique<juce::AudioParameterBool>("arpOn", "Arpeggiator On", false));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("arpRate", "Arp Rate", arpRateChoices, 2));
        params.push_back(std::make_unique<juce::AudioParameterChoice>("arpMode", "Arp Mode", arpModeChoices, 0));
        params.push_back(std::make_unique<juce::AudioParameterInt>("arpOctaves", "Arp Octaves", 1, 3, 2));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("arpGate", "Arp Gate", 0.1f, 0.95f, 0.75f));

        // Compressor
        params.push_back(std::make_unique<juce::AudioParameterBool>("compOn", "Compressor On", true));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("compThresh", "Comp Threshold (dB)", -40.0f, 0.0f, -15.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("compRatio", "Comp Ratio", 1.0f, 12.0f, 3.2f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("compAttack", "Comp Attack (ms)", 0.5f, 100.0f, 15.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("compRelease", "Comp Release (ms)", 10.0f, 500.0f, 140.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("compMakeup", "Comp Makeup (dB)", 0.0f, 18.0f, 2.5f));

        // Delay
        params.push_back(std::make_unique<juce::AudioParameterFloat>("delayTime", "Delay Time (ms)", 20.0f, 1200.0f, 420.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("delayFeedback", "Delay Feedback", 0.0f, 0.85f, 0.42f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("delayMix", "Delay Mix", 0.0f, 1.0f, 0.28f));

        // Reverb
        params.push_back(std::make_unique<juce::AudioParameterFloat>("revSize", "Reverb Size", 0.0f, 1.0f, 0.78f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("revDamp", "Reverb Damping", 0.0f, 1.0f, 0.35f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>("revMix", "Reverb Mix", 0.0f, 1.0f, 0.38f));

        // Master Volume
        params.push_back(std::make_unique<juce::AudioParameterFloat>("masterGain", "Master Volume", 0.0f, 1.5f, 0.85f));

        return { params.begin(), params.end() };
    }

    void selectBuiltInWavetable(int index)
    {
        if (index >= 0 && index < WavetableBank::numBuiltInPresets)
        {
            lastPresetIdx = index;
            customWavetable.reset();
            loadedWavetablePath.clear();
            applyWavetableToAllVoices(builtInTables[index]);
        }
        else if (index == WavetableBank::numBuiltInPresets && customWavetable != nullptr)
        {
            lastPresetIdx = index;
            applyWavetableToAllVoices(customWavetable);
        }
    }

    bool loadWavetableFile(const juce::File& file)
    {
        auto newTable = WavetableBank::loadFromWavFile(file);
        if (newTable != nullptr)
        {
            customWavetable = newTable;
            loadedWavetablePath = file.getFullPathName();
            lastPresetIdx = WavetableBank::numBuiltInPresets;
            applyWavetableToAllVoices(customWavetable);

            if (auto* p = apvts.getParameter("wtPreset"))
                p->setValueNotifyingHost(p->convertTo0to1(static_cast<float>(WavetableBank::numBuiltInPresets)));

            apvts.getParameter("wtOn")->setValueNotifyingHost(1.0f);
            return true;
        }
        return false;
    }

    std::shared_ptr<WavetableData> getActiveWavetable() const
    {
        return std::atomic_load(&activeWavetable);
    }

    float getCurrentEffectiveWtPos() const
    {
        return currentEffectiveWtPos.load();
    }

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

        oversampler = std::make_unique<juce::dsp::Oversampling<float>>(
            2, 1, juce::dsp::Oversampling<float>::filterHalfBandPolyphaseIIR, true);
        oversampler->initProcessing(static_cast<size_t>(samplesPerBlock));

        double oversampledRate = sampleRate * 2.0;
        int oversampledBlockSize = samplesPerBlock * 2;

        for (int p = 0; p < PolySynthVoice::maxPolyVoices; ++p)
            polyVoices[p].prepare(oversampledRate, oversampledBlockSize, p);

        applyWavetableToAllVoices(getActiveWavetable());

        juce::dsp::ProcessSpec stereoSpec { sampleRate, static_cast<juce::uint32>(samplesPerBlock), 2 };
        juce::dsp::ProcessSpec monoSpec   { sampleRate, static_cast<juce::uint32>(samplesPerBlock), 1 };

        compressor.prepare(stereoSpec);
        compressor.reset();

        delayLineL.prepare(monoSpec);
        delayLineR.prepare(monoSpec);
        delayLineL.setMaximumDelayInSamples(static_cast<int>(sampleRate * 2.0));
        delayLineR.setMaximumDelayInSamples(static_cast<int>(sampleRate * 2.0));
        delayLineL.reset();
        delayLineR.reset();

        reverb.setSampleRate(sampleRate);
        reverb.reset();

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

        int presetIdx = static_cast<int>(*apvts.getRawParameterValue("wtPreset"));
        if (presetIdx != lastPresetIdx)
            selectBuiltInWavetable(presetIdx);

        double bpm = 120.0;
        if (auto* ph = getPlayHead())
        {
            if (auto pos = ph->getPosition())
                if (auto hostBpm = pos->getBpm())
                    if (*hostBpm > 10.0) bpm = *hostBpm;
        }

        const int maxAllowedPoly = juce::jlimit(1, PolySynthVoice::maxPolyVoices,
            static_cast<int>(*apvts.getRawParameterValue("polyVoices")));

        // 1. Polyphones MIDI Input Handling
        for (const auto metadata : midiMessages)
        {
            const auto msg = metadata.getMessage();
            if (msg.isNoteOn())
            {
                int note = msg.getNoteNumber();
                heldNotes.erase(std::remove(heldNotes.begin(), heldNotes.end(), note), heldNotes.end());
                heldNotes.push_back(note);

                if (*apvts.getRawParameterValue("arpOn") < 0.5f)
                    triggerPolyNoteOn(note, msg.getFloatVelocity(), maxAllowedPoly);
            }
            else if (msg.isNoteOff())
            {
                int note = msg.getNoteNumber();
                heldNotes.erase(std::remove(heldNotes.begin(), heldNotes.end(), note), heldNotes.end());

                if (*apvts.getRawParameterValue("arpOn") < 0.5f)
                {
                    triggerPolyNoteOff(note);
                    if (maxAllowedPoly == 1 && !heldNotes.empty())
                        triggerPolyNoteOn(heldNotes.back(), 0.85f, 1);
                }
            }
            else if (msg.isAllNotesOff() || msg.isAllSoundOff())
            {
                heldNotes.clear();
                for (auto& v : polyVoices)
                    v.stopNote();
            }
        }

        // 2. Arpeggiator
        bool arpEnabled = *apvts.getRawParameterValue("arpOn") > 0.5f;
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
                int rateIdx = static_cast<int>(*apvts.getRawParameterValue("arpRate"));
                double div = 4.0;
                if (rateIdx == 0) div = 1.0;
                else if (rateIdx == 1) div = 2.0;
                else if (rateIdx == 2) div = 3.0;
                else if (rateIdx == 3) div = 4.0;
                else if (rateIdx == 4) div = 8.0;

                double samplesPerStep = (currentSampleRate * 60.0) / (bpm * div);
                double gateRatio = *apvts.getRawParameterValue("arpGate");

                arpCounter += numSamples;
                if (arpCounter >= samplesPerStep || !arpNoteActive)
                {
                    if (arpCounter >= samplesPerStep)
                        arpCounter -= samplesPerStep;

                    std::vector<int> sortedNotes = heldNotes;
                    std::sort(sortedNotes.begin(), sortedNotes.end());

                    int octaves = static_cast<int>(*apvts.getRawParameterValue("arpOctaves"));
                    std::vector<int> pattern;
                    for (int o = 0; o < octaves; ++o)
                        for (int n : sortedNotes)
                            pattern.push_back(juce::jlimit(0, 127, n + o * 12));

                    int mode = static_cast<int>(*apvts.getRawParameterValue("arpMode"));
                    int patSize = static_cast<int>(pattern.size());

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
                        if (arpStep >= patSize)
                        {
                            arpStep = std::max(0, patSize - 2);
                            arpDirection = -1;
                        }
                        else if (arpStep < 0)
                        {
                            arpStep = std::min(patSize - 1, 1);
                            arpDirection = 1;
                        }
                    }

                    int trigNote = pattern[static_cast<size_t>(arpStep)];
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

        // 3. Wavetable LFO & Globale Parameter berechnen
        float wtLfoRate = *apvts.getRawParameterValue("wtLfoRate");
        int wtLfoWave   = static_cast<int>(*apvts.getRawParameterValue("wtLfoWave"));
        float wtLfoAmt  = *apvts.getRawParameterValue("wtLfoAmt");
        float lfoModVal = wtLfo.processBlock(numSamples, wtLfoRate, wtLfoWave) * wtLfoAmt * 0.5f;

        juce::ADSR::Parameters ampParams {
            *apvts.getRawParameterValue("aAttack"),
            *apvts.getRawParameterValue("aDecay"),
            *apvts.getRawParameterValue("aSustain"),
            *apvts.getRawParameterValue("aRelease")
        };

        juce::ADSR::Parameters filtParams {
            *apvts.getRawParameterValue("fAttack"),
            *apvts.getRawParameterValue("fDecay"),
            *apvts.getRawParameterValue("fSustain"),
            *apvts.getRawParameterValue("aRelease")
        };

        const int numUnison = juce::jlimit(1, UnisonSubVoice::maxUnison,
            static_cast<int>(*apvts.getRawParameterValue("unisonVoices")));
        const double maxUnisonDetune = static_cast<double>(*apvts.getRawParameterValue("unisonDetune"));
        const float unisonSpread = *apvts.getRawParameterValue("unisonSpread");

        std::array<double, UnisonSubVoice::maxUnison> voiceDetuneCents {};
        std::array<float, UnisonSubVoice::maxUnison> voicePanL {};
        std::array<float, UnisonSubVoice::maxUnison> voicePanR {};

        if (numUnison == 1)
        {
            voiceDetuneCents[0] = 0.0;
            voicePanL[0] = 0.70710678f;
            voicePanR[0] = 0.70710678f;
        }
        else
        {
            for (int u = 0; u < numUnison; ++u)
            {
                double normPos = (2.0 * static_cast<double>(u) / static_cast<double>(numUnison - 1)) - 1.0;
                double curvedDetune = std::copysign(std::pow(std::abs(normPos), 1.15), normPos);
                voiceDetuneCents[u] = curvedDetune * maxUnisonDetune;

                float panAngle = static_cast<float>((normPos * unisonSpread + 1.0) * 0.25 * juce::MathConstants<double>::pi);
                voicePanL[u] = std::cos(panAngle);
                voicePanR[u] = std::sin(panAngle);
            }
        }

        const float unisonGainComp = 1.0f / std::sqrt(static_cast<float>(numUnison));
        const float polyGainComp = 0.52f;

        bool oscOn[3] = {
            *apvts.getRawParameterValue("osc1On") > 0.5f,
            *apvts.getRawParameterValue("osc2On") > 0.5f,
            *apvts.getRawParameterValue("osc3On") > 0.5f
        };
        int oscWave[3] = {
            static_cast<int>(*apvts.getRawParameterValue("osc1Wave")),
            static_cast<int>(*apvts.getRawParameterValue("osc2Wave")),
            static_cast<int>(*apvts.getRawParameterValue("osc3Wave"))
        };
        int oscOct[3] = {
            static_cast<int>(*apvts.getRawParameterValue("osc1Oct")),
            static_cast<int>(*apvts.getRawParameterValue("osc2Oct")),
            static_cast<int>(*apvts.getRawParameterValue("osc3Oct"))
        };
        float oscDetune[3] = {
            *apvts.getRawParameterValue("osc1Detune"),
            *apvts.getRawParameterValue("osc2Detune"),
            *apvts.getRawParameterValue("osc3Detune")
        };
        float oscLevel[3] = {
            *apvts.getRawParameterValue("osc1Level"),
            *apvts.getRawParameterValue("osc2Level"),
            *apvts.getRawParameterValue("osc3Level")
        };

        bool wtOn     = *apvts.getRawParameterValue("wtOn") > 0.5f;
        int wtOct     = static_cast<int>(*apvts.getRawParameterValue("wtOct"));
        float wtDet   = *apvts.getRawParameterValue("wtDetune");
        float wtPos   = *apvts.getRawParameterValue("wtPos");
        float wtEnv   = *apvts.getRawParameterValue("wtEnvAmt");
        float wtLevel = *apvts.getRawParameterValue("wtLevel");

        const double octMults[5] = { 0.25, 0.5, 1.0, 2.0, 4.0 };
        double osSampleRate = currentSampleRate * 2.0;
        float glideMs = *apvts.getRawParameterValue("glide");
        double glideCoeff = (glideMs <= 0.5f) ? 1.0 : (1.0 - std::exp(-1.0 / (osSampleRate * (glideMs * 0.001))));
        double driftAmountCents = static_cast<double>(*apvts.getRawParameterValue("analogDrift"));
        const double oscDriftScale[3] = { 0.85, 1.05, 1.20 };

        float drive = *apvts.getRawParameterValue("drive");
        float driveNorm = 1.0f / std::tanh(drive);
        float baseCutoff = *apvts.getRawParameterValue("cutoff");
        float resonance = *apvts.getRawParameterValue("resonance");
        float contour = *apvts.getRawParameterValue("contour");

        // 4. Polyphone Stimmen im 2x Oversampling-Raum rendern & summieren
        juce::dsp::AudioBlock<float> stereoBlock(buffer);
        auto osBlock = oversampler->processSamplesUp(stereoBlock);
        const int osNumSamples = static_cast<int>(osBlock.getNumSamples());

        juce::AudioBuffer<float> voiceOsBuffer(2, osNumSamples);
        juce::dsp::AudioBlock<float> voiceOsBlock(voiceOsBuffer);
        auto voiceLeftBlock  = voiceOsBlock.getSingleChannelBlock(0);
        auto voiceRightBlock = voiceOsBlock.getSingleChannelBlock(1);

        auto* outOsL = osBlock.getChannelPointer(0);
        auto* outOsR = osBlock.getChannelPointer(1);

        float latestWtPosForGui = juce::jlimit(0.0f, 1.0f, wtPos + lfoModVal);
        bool anyVoiceActive = false;

        for (int p = 0; p < PolySynthVoice::maxPolyVoices; ++p)
        {
            auto& pv = polyVoices[p];
            if (!pv.active)
                continue;

            anyVoiceActive = true;
            pv.ampEnv.setParameters(ampParams);
            pv.filterEnv.setParameters(filtParams);

            auto* vL = voiceOsBuffer.getWritePointer(0);
            auto* vR = voiceOsBuffer.getWritePointer(1);

            float fEnvVal = 0.0f;
            for (int s = 0; s < osNumSamples; ++s)
                fEnvVal = pv.filterEnv.getNextSample();

            float effectiveWtPos = juce::jlimit(0.0f, 1.0f, wtPos + fEnvVal * wtEnv + lfoModVal);
            latestWtPosForGui = effectiveWtPos;

            for (int s = 0; s < osNumSamples; ++s)
            {
                pv.currentFreq += (pv.targetFreq - pv.currentFreq) * glideCoeff;

                float sumL = 0.0f;
                float sumR = 0.0f;

                for (int u = 0; u < numUnison; ++u)
                {
                    auto& sub = pv.subVoices[u];
                    double uDetune = voiceDetuneCents[u];
                    float subSample = 0.0f;

                    for (int o = 0; o < 3; ++o)
                    {
                        double currentDrift = sub.driftGens[o].getNextDriftCents(driftAmountCents * oscDriftScale[o]);
                        if (!oscOn[o]) continue;

                        sub.oscs[o].setWaveform(oscWave[o]);
                        double totalDetuneCents = static_cast<double>(oscDetune[o]) + uDetune + currentDrift;
                        double detuneFactor = std::pow(2.0, totalDetuneCents / 1200.0);

                        sub.oscs[o].setFrequency(pv.currentFreq * octMults[oscOct[o]] * detuneFactor);
                        subSample += sub.oscs[o].processSample() * oscLevel[o];
                    }

                    double wtDrift = sub.wtDriftGen.getNextDriftCents(driftAmountCents * 0.5);
                    if (wtOn)
                    {
                        double wtDetFactor = std::pow(2.0, (static_cast<double>(wtDet) + uDetune + wtDrift) / 1200.0);
                        sub.wtOsc.setFrequency(pv.currentFreq * octMults[wtOct] * wtDetFactor);
                        subSample += sub.wtOsc.processSample(effectiveWtPos) * wtLevel;
                    }

                    sumL += subSample * voicePanL[u];
                    sumR += subSample * voicePanR[u];
                }

                float envVal = pv.ampEnv.getNextSample() * pv.velocity;
                float sL = (std::tanh(sumL * unisonGainComp * drive) * driveNorm) * envVal;
                float sR = (std::tanh(sumR * unisonGainComp * drive) * driveNorm) * envVal;

                vL[s] = sL;
                vR[s] = sR;
            }

            float modCutoff = juce::jlimit(20.0f, 18500.0f, baseCutoff + fEnvVal * contour * 12000.0f);
            pv.ladderFilterL.setCutoffFrequencyHz(modCutoff);
            pv.ladderFilterR.setCutoffFrequencyHz(modCutoff);
            pv.ladderFilterL.setResonance(resonance);
            pv.ladderFilterR.setResonance(resonance);

            pv.ladderFilterL.process(juce::dsp::ProcessContextReplacing<float>(voiceLeftBlock));
            pv.ladderFilterR.process(juce::dsp::ProcessContextReplacing<float>(voiceRightBlock));

            for (int s = 0; s < osNumSamples; ++s)
            {
                outOsL[s] += vL[s] * polyGainComp;
                outOsR[s] += vR[s] * polyGainComp;
            }

            if (!pv.ampEnv.isActive())
                pv.active = false;
        }

        if (!anyVoiceActive)
            latestWtPosForGui = juce::jlimit(0.0f, 1.0f, wtPos + lfoModVal);

        currentEffectiveWtPos.store(latestWtPosForGui);

        oversampler->processSamplesDown(stereoBlock);

        auto* leftPtr  = buffer.getWritePointer(0);
        auto* rightPtr = buffer.getWritePointer(1);

        // 5. VCA Compressor
        if (*apvts.getRawParameterValue("compOn") > 0.5f)
        {
            compressor.setThreshold(*apvts.getRawParameterValue("compThresh"));
            compressor.setRatio(*apvts.getRawParameterValue("compRatio"));
            compressor.setAttack(*apvts.getRawParameterValue("compAttack"));
            compressor.setRelease(*apvts.getRawParameterValue("compRelease"));

            compressor.process(juce::dsp::ProcessContextReplacing<float>(stereoBlock));

            float makeupGain = juce::Decibels::decibelsToGain(apvts.getRawParameterValue("compMakeup")->load());
            buffer.applyGain(makeupGain);
        }

        // 6. Stereo Ping-Pong / Dual Delay
        float delayMs = *apvts.getRawParameterValue("delayTime");
        float delayFb = *apvts.getRawParameterValue("delayFeedback");
        float delayMix = *apvts.getRawParameterValue("delayMix");

        if (delayMix > 0.001f)
        {
            float delaySamplesL = static_cast<float>((delayMs * 0.001) * currentSampleRate);
            float delaySamplesR = static_cast<float>((delayMs * 0.001 * 1.12) * currentSampleRate);

            for (int i = 0; i < numSamples; ++i)
            {
                float inL = leftPtr[i];
                float inR = rightPtr[i];

                float delL = delayLineL.popSample(0, delaySamplesL, true);
                float delR = delayLineR.popSample(0, delaySamplesR, true);

                delayLineL.pushSample(0, inL + delR * delayFb);
                delayLineR.pushSample(0, inR + delL * delayFb);

                leftPtr[i]  = inL * (1.0f - delayMix * 0.5f) + delL * delayMix;
                rightPtr[i] = inR * (1.0f - delayMix * 0.5f) + delR * delayMix;
            }
        }

        // 7. Stereo Reverb
        float revMix = *apvts.getRawParameterValue("revMix");
        if (revMix > 0.001f)
        {
            juce::Reverb::Parameters rp;
            rp.roomSize = *apvts.getRawParameterValue("revSize");
            rp.damping  = *apvts.getRawParameterValue("revDamp");
            rp.wetLevel = revMix;
            rp.dryLevel = 1.0f - (revMix * 0.4f);
            rp.width    = 1.0f;
            reverb.setParameters(rp);
            reverb.processStereo(leftPtr, rightPtr, numSamples);
        }

        // 8. Master Volume & Safety Limiter
        float master = *apvts.getRawParameterValue("masterGain");
        for (int ch = 0; ch < 2; ++ch)
        {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
            {
                float v = data[i] * master;
                data[i] = juce::jlimit(-1.0f, 1.0f, v);
            }
        }
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
            juce::String path = tree.getProperty("loadedWavetablePath", "").toString();
            apvts.replaceState(tree);
            if (path.isNotEmpty())
                loadWavetableFile(juce::File(path));
        }
    }

    juce::AudioProcessorValueTreeState apvts;

private:
    void applyWavetableToAllVoices(std::shared_ptr<WavetableData> table)
    {
        if (table == nullptr) return;
        std::atomic_store(&activeWavetable, table);
        for (auto& pv : polyVoices)
            for (auto& sub : pv.subVoices)
                sub.wtOsc.setTable(table);
    }

    void triggerPolyNoteOn(int midiNote, float velocity, int maxAllowedPoly)
    {
        globalNoteCounter++;
        double newFreq = juce::MidiMessage::getMidiNoteInHertz(midiNote);

        for (int i = maxAllowedPoly; i < PolySynthVoice::maxPolyVoices; ++i)
            polyVoices[i].active = false;

        for (int i = 0; i < maxAllowedPoly; ++i)
        {
            if (polyVoices[i].active && polyVoices[i].midiNote == midiNote)
            {
                polyVoices[i].startNote(midiNote, velocity, lastTriggeredFreq, maxAllowedPoly == 1, globalNoteCounter);
                lastTriggeredFreq = newFreq;
                return;
            }
        }

        for (int i = 0; i < maxAllowedPoly; ++i)
        {
            if (!polyVoices[i].active)
            {
                polyVoices[i].startNote(midiNote, velocity, lastTriggeredFreq, maxAllowedPoly == 1, globalNoteCounter);
                lastTriggeredFreq = newFreq;
                return;
            }
        }

        int oldestIdx = 0;
        uint64_t oldestAge = polyVoices[0].noteOnAge;
        for (int i = 1; i < maxAllowedPoly; ++i)
        {
            if (polyVoices[i].noteOnAge < oldestAge)
            {
                oldestAge = polyVoices[i].noteOnAge;
                oldestIdx = i;
            }
        }

        polyVoices[oldestIdx].startNote(midiNote, velocity, lastTriggeredFreq, maxAllowedPoly == 1, globalNoteCounter);
        lastTriggeredFreq = newFreq;
    }

    void triggerPolyNoteOff(int midiNote)
    {
        for (auto& pv : polyVoices)
            if (pv.active && pv.midiNote == midiNote)
                pv.stopNote();
    }

    double currentSampleRate = 44100.0;
    double lastTriggeredFreq = 110.0;
    uint64_t globalNoteCounter = 0;

    WavetableLFO wtLfo;
    std::array<PolySynthVoice, PolySynthVoice::maxPolyVoices> polyVoices;
    std::array<std::shared_ptr<WavetableData>, WavetableBank::numBuiltInPresets> builtInTables;
    std::shared_ptr<WavetableData> customWavetable;
    std::shared_ptr<WavetableData> activeWavetable;
    juce::String loadedWavetablePath;
    int lastPresetIdx = 0;
    std::atomic<float> currentEffectiveWtPos { 0.45f };

    std::unique_ptr<juce::dsp::Oversampling<float>> oversampler;
    juce::dsp::Compressor<float> compressor;
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> delayLineL;
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> delayLineR;
    juce::Reverb reverb;

    std::vector<int> heldNotes;
    double arpCounter = 0.0;
    int arpStep = 0;
    int arpDirection = 1;
    bool arpNoteActive = false;
    int lastArpNote = 60;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModelD3AudioProcessor)
};

// ==============================================================================
// 2D Wavetable Visualizer Component (Echtzeit-Vorschau aller Frames + aktiver Frame)
// ==============================================================================
class Wavetable2DDisplay : public juce::Component, private juce::Timer
{
public:
    explicit Wavetable2DDisplay(ModelD3AudioProcessor& p) : processor(p)
    {
        startTimerHz(30);
    }

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
        float morphPos = processor.getCurrentEffectiveWtPos();

        g.setColour(juce::Colour(0xff8fa3bf));
        g.setFont(juce::Font(12.0f, juce::Font::bold));
        juce::String title = (table != nullptr) ? table->name : "Keine Wavetable";
        g.drawText(title, headerArea.removeFromLeft(headerArea.getWidth() * 0.72f), juce::Justification::centredLeft, true);

        g.setColour(juce::Colour(0xff4da6ff));
        g.drawText("Pos: " + juce::String(static_cast<int>(morphPos * 100.0f)) + "%", headerArea, juce::Justification::centredRight, false);

        if (table == nullptr || table->frames.empty())
            return;

        const int numFrames = static_cast<int>(table->frames.size());
        const int activeFrameIdx = juce::jlimit(0, numFrames - 1, static_cast<int>(morphPos * static_cast<float>(numFrames - 1)));

        g.setColour(juce::Colour(0xff1f2630));
        g.drawHorizontalLine(static_cast<int>(area.getCentreY()), area.getX(), area.getRight());

        int step = std::max(1, numFrames / 16);
        for (int f = 0; f < numFrames; f += step)
        {
            if (f == activeFrameIdx) continue;
            float dist = std::abs(static_cast<float>(f - activeFrameIdx)) / static_cast<float>(std::max(1, numFrames));
            float alpha = juce::jlimit(0.06f, 0.24f, 0.26f * (1.0f - dist));
            drawWaveformPath(g, table->frames[f], area, juce::Colour(0xff4da6ff).withAlpha(alpha), 1.0f, (f - activeFrameIdx) * 0.12f);
        }

        drawWaveformPath(g, table->frames[activeFrameIdx], area, juce::Colour(0xffffaa33), 2.2f, 0.0f);
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
            float normX = static_cast<float>(i) / static_cast<float>(numPoints - 1);
            int idx = juce::jlimit(0, tableLen - 1, static_cast<int>(normX * static_cast<float>(tableLen - 1)));
            float sample = juce::jlimit(-1.05f, 1.05f, frame[idx]);

            float x = area.getX() + normX * area.getWidth();
            float y = area.getCentreY() - sample * (area.getHeight() * 0.44f) + yShift;

            if (i == 0) path.startNewSubPath(x, y);
            else        path.lineTo(x, y);
        }

        g.setColour(col);
        g.strokePath(path, juce::PathStrokeType(thickness));
    }

    void timerCallback() override
    {
        repaint();
    }

    ModelD3AudioProcessor& processor;
};

// ==============================================================================
// Custom Editor mit 2D Wavetable Fenster + 50er Preset-Auswahl + WAV-Loader
// ==============================================================================
class ModelD3AudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit ModelD3AudioProcessorEditor(ModelD3AudioProcessor& p)
        : AudioProcessorEditor(&p), audioProcessor(p), wtDisplay(p)
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

        static const std::vector<SectionDef> sectionDefs = {
            { "OSC 1", 0, { "osc1On", "osc1Wave", "osc1Oct", "osc1Detune", "osc1Level" } },
            { "OSC 2", 0, { "osc2On", "osc2Wave", "osc2Oct", "osc2Detune", "osc2Level" } },
            { "OSC 3", 0, { "osc3On", "osc3Wave", "osc3Oct", "osc3Detune", "osc3Level" } },
            { "WAVETABLE OSC", 0, { "wtOn", "wtOct", "wtDetune", "wtPos", "wtEnvAmt", "wtLfoWave", "wtLfoRate", "wtLfoAmt", "wtLevel" } },
            { "FILTER", 0, { "cutoff", "resonance", "contour" } },
            { "FILTER ENV", 0, { "fAttack", "fDecay", "fSustain" } },
            { "AMP ENV", 0, { "aAttack", "aDecay", "aSustain", "aRelease" } },
            { "VOICE", 1, { "polyVoices", "unisonVoices", "unisonDetune", "unisonSpread", "analogDrift", "glide" } },
            { "MIXER", 1, { "drive" } },
            { "ARPEGGIATOR", 1, { "arpOn", "arpRate", "arpMode", "arpOctaves", "arpGate" } },
            { "COMPRESSOR", 1, { "compOn", "compThresh", "compRatio", "compAttack", "compRelease", "compMakeup" } },
            { "DELAY", 1, { "delayTime", "delayFeedback", "delayMix" } },
            { "REVERB", 1, { "revSize", "revDamp", "revMix" } },
            { "MASTER", 1, { "masterGain" } }
        };

        juce::StringArray placed;
        for (const auto& def : sectionDefs)
        {
            Section s;
            s.title = def.title;
            s.column = def.column;
            s.header = std::make_unique<juce::Label>(juce::String(), def.title);
            s.header->setFont(juce::Font(juce::FontOptions(12.0f)));
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
            sections.push_back(std::move(s));
        }

        Section other;
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
                other.header->setFont(juce::Font(juce::FontOptions(12.0f)));
                other.header->setColour(juce::Label::textColourId, juce::Colour(0xff6fb3e8));
                addAndMakeVisible(*other.header);
            }
            addControlToSection(other, param);
        }
        if (other.header != nullptr)
            sections.push_back(std::move(other));

        const int contentBottom = layoutSections(kEditorWidth, 178);
        setResizable(true, true);
        setResizeLimits(kEditorWidth, contentBottom + 10, 2000, 3000);
        setSize(kEditorWidth, contentBottom + 10);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xff181b20));
        g.setColour(juce::Colour(0xff232730));
        g.fillRect(0, 0, getWidth(), 168);

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
        auto topPanel = area.removeFromTop(168).reduced(10, 8);

        auto controlRow = topPanel.removeFromTop(30);
        presetBox.setBounds(controlRow.removeFromLeft(controlRow.getWidth() - 125));
        controlRow.removeFromLeft(8);
        loadBtn.setBounds(controlRow);

        topPanel.removeFromTop(8);
        wtDisplay.setBounds(topPanel);

        layoutSections(getWidth(), 178);
    }

private:
    static constexpr int kEditorWidth = 960;

    struct Control
    {
        std::unique_ptr<juce::Component> comp;
        std::unique_ptr<juce::Label> label;
        bool isKnob = false;
    };

    struct SectionDef
    {
        juce::String title;
        int column;
        std::vector<juce::String> ids;
    };

    struct Section
    {
        juce::String title;
        int column = 0;
        juce::Rectangle<int> area;
        std::unique_ptr<juce::Label> header;
        std::vector<Control> controls;
    };

    void addControlToSection(Section& s, juce::AudioProcessorParameter* param)
    {
        auto* withID = dynamic_cast<juce::AudioProcessorParameterWithID*>(param);

        Control c;
        c.label = std::make_unique<juce::Label>(juce::String(), param->getName(64));
        c.label->setJustificationType(juce::Justification::centred);
        c.label->setColour(juce::Label::textColourId, juce::Colours::lightgrey);
        c.label->setFont(juce::Font(juce::FontOptions(11.0f)));
        addAndMakeVisible(*c.label);

        const auto id = withID != nullptr ? withID->paramID : juce::String();

        if (auto* choice = dynamic_cast<juce::AudioParameterChoice*>(param))
        {
            auto combo = std::make_unique<juce::ComboBox>();
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
        }

        addAndMakeVisible(*c.comp);
        s.controls.push_back(std::move(c));
    }

    int layoutSections(int width, int top)
    {
        const int margin = 12;
        const int gap = 12;
        const int colW = juce::jmax(240, (width - 2 * margin - gap) / 2);
        const int ctrlH = 88;
        const int labelH = 18;
        const int rowH = ctrlH + labelH + 8;
        const int headerH = 24;
        const int sectionGap = 12;

        int colY[2] = { top, top };
        const int colX[2] = { margin, margin + colW + gap };

        for (auto& s : sections)
        {
            const int n = static_cast<int>(s.controls.size());
            if (n == 0)
                continue;

            const int col = juce::jlimit(0, 1, s.column);
            const int colsPerRow = juce::jlimit(3, 5, n);
            const int rows = (n + colsPerRow - 1) / colsPerRow;
            const int h = headerH + rows * rowH + 6;

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
                if (ctl.isKnob)
                    ctl.comp->setBounds(cx + 4, cy, cellW - 8, ctrlH);
                else
                    ctl.comp->setBounds(cx + 4, cy + (ctrlH - 30) / 2, cellW - 8, 30);
                ctl.label->setBounds(cx + 4, cy + ctrlH + 2, cellW - 8, labelH);
            }

            colY[col] = y + rows * rowH + sectionGap;
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
    juce::ComboBox presetBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> presetAttachment;
    juce::TextButton loadBtn;
    std::unique_ptr<juce::FileChooser> fileChooser;
    std::vector<Section> sections;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>> buttonAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>> comboAttachments;
};

juce::AudioProcessorEditor* ModelD3AudioProcessor::createEditor()
{
    return new ModelD3AudioProcessorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ModelD3AudioProcessor();
}
