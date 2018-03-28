package edu.utdallas.hltri.inquire.engines;

import java.util.List;

import edu.utdallas.hltri.inquire.SearchResult;

/**
 * Created by travis on 5/31/17.
 */
@SuppressWarnings("unused")
public interface SearchResults<T, R extends SearchResult<T>> {

  double getMaxScore();

  int getTotalHits();

  List<R> getResults();

}
