package edu.utdallas.hltri.inquire.lucene;

import edu.utdallas.hltri.scribe.Stopwords;
import org.apache.lucene.analysis.en.EnglishAnalyzer;

@SuppressWarnings("unused")
public class LuceneStopwords extends Stopwords {
  public static final long serialVersionUID = 1L;

  public static final Stopwords LUCENE = new LuceneStopwords();

  private LuceneStopwords() {
    super();
    for (Object word : EnglishAnalyzer.getDefaultStopSet()) {
      delegate.add(new String((char[]) word));
    }
  }
}
