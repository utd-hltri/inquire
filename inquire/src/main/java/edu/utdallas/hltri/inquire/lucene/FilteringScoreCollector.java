package edu.utdallas.hltri.inquire.lucene;

import it.unimi.dsi.fastutil.ints.Int2FloatMap;
import it.unimi.dsi.fastutil.ints.Int2FloatOpenHashMap;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.SimpleCollector;

import java.io.IOException;
import java.util.BitSet;

public class FilteringScoreCollector extends SimpleCollector {
  private final Int2FloatMap scores = new Int2FloatOpenHashMap();
  private final BitSet filter;

  private Scorer scorer;
  int contextBaseId;

  public FilteringScoreCollector(BitSet filter) {
    this.filter = filter;
  }

  @Override
  protected void doSetNextReader(LeafReaderContext context) throws IOException {
    this.contextBaseId = context.docBase;
  }

  @Override
  public void setScorer(Scorer scorer) throws IOException {
    this.scorer = scorer;
  }

  @Override
  public void collect(int contextDocId) throws IOException {
    final int globalId = contextBaseId + contextDocId;
    if (filter.get(globalId)) {
      scores.put(globalId, scorer.score());
    }
  }

  @Override
  public boolean needsScores() {
    return true;
  }

  public Int2FloatMap getScores() {
    return scores;
  }
}
