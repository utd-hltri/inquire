package edu.utdallas.hltri.inquire.ie.filter;

import edu.utdallas.hltri.inquire.engines.PMCSearchEngine;
import edu.utdallas.hltri.conf.Config;

/**
 * Created by travis on 7/11/14.
 */
public class PMCKeywordFilter extends OccurrenceFilter {

  private static final Config conf = Config.load("inquire.pubmed");

  public PMCKeywordFilter() {
    this(conf.getInt("threshold"));
  }

  public PMCKeywordFilter(int threshold) {
    super(threshold, new PMCSearchEngine(), PMCSearchEngine.TEXT_FIELD);
  }
}
