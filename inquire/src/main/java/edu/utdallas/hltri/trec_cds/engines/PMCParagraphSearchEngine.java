package edu.utdallas.hltri.trec_cds.engines;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.Lists;

import edu.utdallas.hltri.inquire.lucene.DocumentFactory;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import edu.utdallas.hltri.conf.Config;
import edu.utdallas.hltri.inquire.engines.PMCSearchEngine;
import edu.utdallas.hltri.inquire.lucene.LuceneSearchEngine;
import edu.utdallas.hltri.logging.Logger;

/**
 *
 * @author travis
 */
public class PMCParagraphSearchEngine extends LuceneSearchEngine<PMCParagraphSearchEngine.PMCParagraph> {

  Logger log = Logger.get(PMCParagraphSearchEngine.class);

  public static final String TEXT_FIELD = "text";
  public static final String PARAGRAPH_ID_FIELD = "pid";
  public static final String UNIQUE_ID_FIELD = "uid";
  public static final String PMC_ID_FIELD = "pmc_id";

  private static final Config conf = Config.load("inquire.pubmed.paragraph");

  public PMCParagraphSearchEngine(String path) {
    super(path,
          new EnglishAnalyzer(),
          TEXT_FIELD,
          DocumentFactory.eager(PMCParagraph::fromLucene));
  }

  public PMCParagraphSearchEngine() {
    this(conf.getString("path"));
  }

  public Query getPmcIdQuery(String pmcid) {
    return new TermQuery(new Term("pmc_id", pmcid));
  }

  public Query getUidQuery(String uid) {
    return new TermQuery(new Term("uid", uid));
  }

  private Collection<PMCParagraph> _getParagraphs(String pmcid) {
    return withSearcher(searcher -> {
      try {
        final TopDocs
            luceneResults =
            searcher.search(new TermQuery(new Term("pmc_id", pmcid)), Integer.MAX_VALUE);
        List<PMCParagraph> results = Lists.newArrayListWithExpectedSize(luceneResults.totalHits);
        for (final ScoreDoc d : luceneResults.scoreDocs) {
          final Document luceneDoc = searcher.getIndexReader().document(d.doc, Collections
              .singleton(PMCParagraphSearchEngine.PMC_ID_FIELD));
          results.add(PMCParagraph.fromLucene(luceneDoc));
        }
        return results;
      } catch (IOException | ArrayIndexOutOfBoundsException ex) {
        throw new PMCSearchEngine.IllegalPMCIDException(pmcid);
      }
    });
  }

  public Collection<PMCParagraph> getParagraphs(String pmcid) {
    return paragraphCache.getUnchecked(pmcid);
  }

  public long getFieldLength(int id) {
    return super.getFieldLength(id, TEXT_FIELD);
  }

  final LoadingCache<String, Collection<PMCParagraph>> paragraphCache = CacheBuilder.newBuilder()
      .concurrencyLevel(Runtime.getRuntime().availableProcessors())
      .maximumSize(1_000)
      .build(CacheLoader.from(this::_getParagraphs));

  public Collection<PMCParagraph> getParagraphs(PMCSearchEngine.PMCResult d) {
    return getParagraphs(d.getPmcId());
  }

  public static class PMCParagraph {
    private final String pmcId;
    private final String uid;
    private final String paragraphId;

    private PMCParagraph(String pmcId, String paragraphId, String uniqueId) {
      this.pmcId = pmcId;
      this.paragraphId = paragraphId;
      this.uid = uniqueId;
    }

    public static PMCParagraph fromLucene(Document d) {
      return new PMCParagraph(d.get(PMC_ID_FIELD),
                              d.get(PARAGRAPH_ID_FIELD),
                              d.get(UNIQUE_ID_FIELD));
    }

    public String getPmcId() {
      return pmcId;
    }

    public String getParagraphId() {
      return paragraphId;
    }

    public String getUid() {
      return uid;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      PMCParagraph that = (PMCParagraph) o;
      return Objects.equals(pmcId, that.pmcId) &&
             Objects.equals(paragraphId, that.paragraphId);
    }

    @Override
    public int hashCode() {
      return Objects.hash(pmcId, paragraphId);
    }
  }
}
