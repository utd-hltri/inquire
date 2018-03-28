package edu.utdallas.hltri.inquire.lucene;

import gnu.trove.map.TIntFloatMap;
import gnu.trove.map.hash.TIntFloatHashMap;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Scorer;

import java.io.IOException;

/**
 * Created by travis on 8/7/14.
 */
public class AllDocsCollector implements Collector {
  private final TIntFloatMap scores = new TIntFloatHashMap();

  @Override
  public LeafCollector getLeafCollector(LeafReaderContext context) throws IOException {
    return new LeafCollector() {
      Scorer scorer;
      final int docBase = context.docBase;

      @Override
      public void setScorer(Scorer scorer) throws IOException {
          this.scorer = scorer;
      }

      @Override
      public void collect(int doc) throws IOException {
        try {
          scores.put(docBase + doc, scorer.score());
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
    };
  }

  @Override
  public boolean needsScores() {
    return true;
  }

  public static AllDocsCollector create() {
    return new AllDocsCollector();
  }

  TIntFloatMap getScores() {
    return scores;
  }

  public float getScore(int doc) {
    return scores.get(doc);
  }
}
