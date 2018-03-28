package edu.utdallas.hltri.inquire.lucene;

import edu.utdallas.hltri.inquire.SearchResult;
import edu.utdallas.hltri.logging.Logger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;

import edu.utdallas.hltri.inquire.engines.SearchResultsList;
import java.util.function.IntConsumer;
import java.util.function.ToDoubleFunction;
import java.util.function.ToIntFunction;


/**
 * Created by travis on 5/31/17.
 */
public class LuceneSearchResultsList<D> extends SearchResultsList<D, LuceneResult<D>> {
  private static final Logger log = Logger.get(LuceneSearchResultsList.class);

  @SuppressWarnings("WeakerAccess")
  public LuceneSearchResultsList(double maxScore, int numHits, List<LuceneResult<D>> resultsList) {
    super(maxScore, numHits, resultsList);
  }

  public static <D> LuceneSearchResultsList<D> wrapToList(
      final Collection<D> resultsList,
      final ToIntFunction<D> rankingFunction,
      final ToDoubleFunction<D> scoringFunction,
      final ToIntFunction<D> luceneIdentifier) {
    final List<LuceneResult<D>> wrappedList = new ArrayList<>(resultsList.size());

    for (final D result : resultsList) {
      if (result instanceof LuceneResult) {
        log.error("Wrapping LuceneResult {} in another LuceneResult!", result);
      }
      wrappedList.add(
          new LuceneResult<>(result,
              scoringFunction.applyAsDouble(result),
              rankingFunction.applyAsInt(result),
              luceneIdentifier.applyAsInt(result)));
    }

    wrappedList.sort(Comparator.comparingInt(SearchResult::getRank));
    final double maxScore = (wrappedList.isEmpty() ? Double.NaN : wrappedList.get(0).score);
    return new LuceneSearchResultsList<>(maxScore, wrappedList.size(), wrappedList);
  }
}
