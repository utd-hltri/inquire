package edu.utdallas.hltri.inquire.engines;

import edu.utdallas.hltri.inquire.lucene.DocumentFactory;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.Query;

import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;

import edu.utdallas.hltri.conf.Config;
import edu.utdallas.hltri.inquire.lucene.LuceneResult;
import edu.utdallas.hltri.inquire.lucene.LuceneSearchEngine;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.struct.Weighted;
import edu.utdallas.hltri.util.Expansion;

/**
 *
 * @author travis
 */
@SuppressWarnings({"WeakerAccess", "unused"})
public class WikiSearchEngine extends LuceneSearchEngine<WikiSearchEngine.WikiArticle> {

  private static final Logger log = Logger.get(WikiSearchEngine.class);

  public static final String TEXT_FIELD = "content";

  private static final Config conf = Config.load("inquire.wikipedia");

  public WikiSearchEngine(String path) {
    super(path,
          new EnglishAnalyzer(),
          TEXT_FIELD,
          DocumentFactory.eager(WikiArticle::fromLucene));
  }

  public static WikiSearchEngine getDefault() {
    return new WikiSearchEngine(conf.getString("path"));
  }

  public Collection<Weighted<String>> searchTitles(Query item) {
    Set<Weighted<String>> results = new TreeSet<>();

    for (LuceneResult<WikiArticle> result : this.search(item, 20).getResults()) {
      results.add(Weighted.create(result.getValue().getTitle().toLowerCase(), result.getScore()));
    }

    Expansion.filterWeightedExpansions(results);

    log.info("Found {} titles for {}", results, item);

    return results;
  }

  @SuppressWarnings({"OptionalUsedAsFieldOrParameterType", "unused"})
  public static class WikiArticle {
    private final String            title;

    public static WikiArticle fromLucene(Document document) {
      return new WikiArticle(document.get(TITLE_FIELD));
    }

    public static WikiArticle orphaned(String title) {
      return new WikiArticle(title);
    }

    public WikiArticle(String title) {
      this.title = title;
    }

    public String getTitle() {
      return title;
    }
  }
}
