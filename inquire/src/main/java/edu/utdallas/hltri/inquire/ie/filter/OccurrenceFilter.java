package edu.utdallas.hltri.inquire.ie.filter;

import org.apache.lucene.search.Query;

import edu.utdallas.hltri.func.CloseablePredicate;
import edu.utdallas.hltri.inquire.lucene.LuceneSearchEngine;
import edu.utdallas.hltri.logging.Logger;

/**
 * Created by travis on 7/14/14.
 */
public class OccurrenceFilter implements CloseablePredicate<CharSequence> {
  private static final Logger log = Logger.get(OccurrenceFilter.class);

  private final int                   threshold;
  private final LuceneSearchEngine<?> engine;
  private final String                field;

  public OccurrenceFilter(int threshold, LuceneSearchEngine<?> engine, String field) {
    this.threshold = threshold;
    this.engine = engine;
    this.field = field;
  }

  @Override public boolean test(CharSequence input) {
    if (input == null) return false;
    if (threshold == 0) return true;
    final Query query = engine.newPhraseQuery(field, input);
    final int hits = engine.getHitCount(query);
    log.debug("|{}| -> {} had {} hits (threshold = {})", input, query, hits, threshold);
    return hits <= threshold && hits >= 1;
  }

  @Override public void close() {

  }
}
