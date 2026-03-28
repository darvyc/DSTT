#include <dstt/mge/synthesiser.hpp>
#include <dstt/utils/math.hpp>

namespace dstt {

size_t SynthesisedOutput::count(Modality m) const {
    size_t c = 0;
    for (const auto& e : elements) {
        if (e.modality == m) ++c;
    }
    return c;
}

double SynthesisedOutput::avg_probability() const {
    if (elements.empty()) return 0.0;
    double s = 0.0;
    for (const auto& e : elements) s += e.probability;
    return s / static_cast<double>(elements.size());
}

Synthesiser::Synthesiser(const Config& cfg) : cfg_(cfg) {}

void Synthesiser::append(OutputElement elem) {
    output_.elements.push_back(std::move(elem));
}

bool Synthesiser::check_consistency() const {
    auto& elems = output_.elements;
    if (elems.size() < 2) return true;

    const auto& prev = elems[elems.size() - 2];
    const auto& curr = elems[elems.size() - 1];

    // Cross-modal check: cosine similarity > 0.7 between embeddings
    // when transitioning between different modalities
    if (prev.modality != curr.modality) {
        if (prev.embedding.size() == curr.embedding.size()) {
            double cs = math::cosine_similarity(prev.embedding, curr.embedding);
            // Threshold from spec: 0.7 for text↔image, relaxed for video
            double threshold = (curr.modality == Modality::Video) ? 0.3 : 0.7;
            return cs > threshold;
        }
    }
    return true;
}

void Synthesiser::reset() {
    output_.elements.clear();
}

} // namespace dstt
