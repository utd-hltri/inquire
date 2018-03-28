package edu.utdallas.hltri.inquire.lucene;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queries.mlt.MoreLikeThis;
import org.apache.lucene.search.Query;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import edu.utdallas.hltri.struct.Weighted;
import edu.utdallas.hltri.util.AbstractExpander;
import edu.utdallas.hltri.util.Expansion;

public class PseudoRelevanceFeedbackExpander extends AbstractExpander<CharSequence, String> implements AutoCloseable {

  private final LuceneSearchEngine<?> searchEngine;
  private final MoreLikeThis mlt;
  private final IndexReader reader;
  private final int numRetrieved;

  public PseudoRelevanceFeedbackExpander(LuceneSearchEngine<?> searchEngine, int numRetrieved) {
    super("PRF");
    this.searchEngine = searchEngine;
    try {
      this.reader = searchEngine.readerManager.acquire();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    this.numRetrieved = numRetrieved;
    this.mlt = new MoreLikeThis(this.reader);
  }

  @Override
  protected Set<String> getExpansions(CharSequence term) {
    final Query query = searchEngine.newParsedQuery(term);
    final Set<String> expansions = getInterestingTerms(query);
    return expansions;
  }

  @Override
  public Expansion<String> expandAll(Iterable<? extends CharSequence> terms) {
    final Query query = searchEngine.newBooleanQuery(Weighted.fixed(terms, 1.0));
    final Set<String> expansions = getInterestingTerms(query);
    return Expansion.newExpansion(this.name, expansions);
  }

  private Set<String> getInterestingTerms(Query query) {
    final Set<String> expansions = new HashSet<>();
    for (LuceneResult<?> result : searchEngine.search(query, this.numRetrieved).getResults()) {
      try {
        final String[] relatedTerms = mlt.retrieveInterestingTerms(result.getLuceneDocId());
        expansions.addAll(Arrays.asList(relatedTerms));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    return expansions;
  }

  @Override
  public void close() {
    try {
      this.reader.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
