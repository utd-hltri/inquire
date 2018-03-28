package edu.utdallas.hltri.inquire.engines;

import edu.utdallas.hltri.conf.Config;
import edu.utdallas.hltri.inquire.lucene.LuceneSearchEngine;
import edu.utdallas.hltri.logging.Logger;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.TermQuery;

import java.io.IOException;
import java.util.Objects;
import java.util.Optional;

/**
 *
 * @author travis
 */
public class PMCSearchEngine extends LuceneSearchEngine<PMCSearchEngine.PMCResult> {

  private static final Logger log = Logger.get(PMCSearchEngine.class);

  @SuppressWarnings("unused")
  public static final String TEXT_FIELD     = "text";

  @SuppressWarnings("unused")
  public static final String KEYWORDS_FIELD = "keywords";

  @SuppressWarnings("unused")
  public static final String ABSTRACT_FIELD = "abstract";

  @SuppressWarnings("unused")
  public static final String TITLE_FIELD    = "article_title";

  private static final Config conf = Config.load("inquire.pubmed");

  @SuppressWarnings("WeakerAccess")
  public PMCSearchEngine(String path) {
    super(path,
          new EnglishAnalyzer(),
          TEXT_FIELD,
          PMCResult::new);
  }

  public PMCSearchEngine() {
    this(conf.getString("path"));
  }

  @SuppressWarnings("unused")
  public int getLuceneId(PMCResult d) {
    return withSearcher(searcher -> {
      try {
        return searcher.search(new TermQuery(new Term("pmc_id", d.getPmcId())), 1).scoreDocs[0].doc;
      } catch (IOException | ArrayIndexOutOfBoundsException ex) {
        throw new IllegalPMCIDException(d.getPmcId());
      }
    });
  }

  @SuppressWarnings("unused")
  public PMCResult getDocumentFromPMCID(String pmcid) {
    return withSearcher(searcher -> {
      try {
        final int luceneId = searcher.search(new TermQuery(new Term("pmc_id", pmcid)), 1).scoreDocs[0].doc;
        final Document luceneDoc = searcher.getIndexReader().document(luceneId);
        return new PMCResult(searcher.getIndexReader(), luceneId);
      } catch (IOException | ArrayIndexOutOfBoundsException ex) {
        throw new IllegalPMCIDException(pmcid);
      }
    });
  }

  public static class IllegalPMCIDException extends IllegalArgumentException {
    public final String id;
    public IllegalPMCIDException(String id) {
      super("Invalid PMCID |" + id + '|');
      this.id = id;
    }
  }

  @SuppressWarnings("unused")
  public PMCResult getDocumentFromPMID(String pmid) {
    return withSearcher(searcher -> {
      try {
        final int luceneId = searcher.search(new TermQuery(new Term("pm_id", pmid)), 1).scoreDocs[0].doc;
        return new PMCResult(searcher.getIndexReader(), luceneId);
      } catch (IOException | ArrayIndexOutOfBoundsException ex) {
        log.error("Unable to find PMID |" + pmid + "|", ex);
        throw new IllegalArgumentException("invalid pmid");
      }
    });
  }

  @SuppressWarnings("unused")
  public static class PMCResult {
    private final Document document;
    private final int luceneId;

    PMCResult(IndexReader reader, int luceneId) {
      this.luceneId = luceneId;
      try {
        this.document = reader.document(luceneId);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    public Optional<Integer> getLuceneId() {
      return Optional.of(luceneId);
    }

    public String getPmcId() {
      return document.get("pmc_id");
    }

    public Optional<String> getPmId() {
      return Optional.ofNullable(document.get("pm_id"));
    }

    public Optional<String> getDoi() {
      return Optional.ofNullable(document.get("doi"));
    }

    public Optional<String> getJournalTitle() {
      return Optional.ofNullable(document.get("journal_title"));
    }

    public String getArticleTitle() {
      return document.get("article_title");
    }

    public String getFilename() {
      return document.get("filename");
    }

    public String getPath() {
      return document.get("path");
    }

    public String getText() { return document.get("text"); }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      PMCResult that = (PMCResult) o;
      return Objects.equals(luceneId, that.luceneId);
    }

    @Override
    public int hashCode() {
      return Objects.hash(luceneId);
    }
  }
}
