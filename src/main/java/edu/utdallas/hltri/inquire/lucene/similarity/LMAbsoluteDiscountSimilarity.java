package edu.utdallas.hltri.inquire.lucene.similarity;

import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.LMSimilarity;

/**
 * Created by travis on 8/7/14.
 */
@SuppressWarnings("unused")
public class LMAbsoluteDiscountSimilarity extends LMSimilarity {
  private final float delta;
  private final long vocab;

  public LMAbsoluteDiscountSimilarity(final float delta, final long vocabularySize) {
    this.delta = delta;
    this.vocab = vocabularySize;
  }

  @Override protected float score(BasicStats stats, float freq, float docLen) {
    return Math.max(freq - delta, 0f) / docLen +
        (delta * vocab / (1 + stats.getTotalTermFreq())) *
            ((LMStats) stats).getCollectionProbability();
  }

  @Override public String getName() {
    return "AbsoluteDiscount(" + delta + "; " + vocab + ")";
  }
}
