package edu.utdallas.hltri.inquire.lucene;

import com.google.common.base.Strings;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;

import com.google.common.collect.Streams;
import gnu.trove.map.TIntFloatMap;

import it.unimi.dsi.fastutil.ints.Int2FloatMap;
import it.unimi.dsi.fastutil.ints.Int2FloatOpenHashMap;

import java.util.Arrays;
import java.util.Comparator;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderManager;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.SearcherFactory;
import org.apache.lucene.search.SearcherManager;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHitCountCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.search.spans.SpanQuery;
import org.apache.lucene.search.spans.SpanWeight;
import org.apache.lucene.search.spans.Spans;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Function;

import javax.annotation.Nonnull;

import edu.utdallas.hltri.inquire.engines.SearchEngine;
import edu.utdallas.hltri.inquire.lucene.similarity.Similarities;
import edu.utdallas.hltri.io.AC;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.struct.Pair;
import edu.utdallas.hltri.struct.Weighted;
import edu.utdallas.hltri.util.Unsafe;

/**
 * Lucene Search Engine
 *
 * @param <D> Type of documents that should be returned
 * @author travis
 */
@SuppressWarnings({"unused", "SameParameterValue", "WeakerAccess"})
public class LuceneSearchEngine<D> implements SearchEngine<Query, D, LuceneResult<D>>, AC {
  /**
   * Version of Lucene library to use
   */
  public static final Version VERSION = Version.LATEST;

  /**
   * Default fields for indexed documents
   */
  public static final String TITLE_FIELD = "title"; // The document's title
  public static final String PATH_FIELD = "path";   // The (file) path to the document

  public static final FieldType TEXT_FIELD_TYPE = new FieldType(TextField.TYPE_STORED);
  static {
    TEXT_FIELD_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    TEXT_FIELD_TYPE.setTokenized(true);
    TEXT_FIELD_TYPE.setStored(true);
    TEXT_FIELD_TYPE.setStoreTermVectors(true);
    TEXT_FIELD_TYPE.setStoreTermVectorPositions(true);
    TEXT_FIELD_TYPE.setStoreTermVectorOffsets(true);
    TEXT_FIELD_TYPE.setStoreTermVectorPayloads(true);
    TEXT_FIELD_TYPE.freeze();
  }

  private static final Logger log = Logger.get(LuceneSearchEngine.class);

  /**
   * Lucene searching components
   */
  protected final Analyzer           analyzer;          // Processes natural language to produce Lucene tokens
  private final   QueryParser        parser;            // Parses a query string to produce a Lucene query
  protected final   ReaderManager      readerManager;     // Allows for multi-threaded index reading
  private final   SearcherManager    searcherManager;   // Allows for multi-threaded index searching
  protected final DocumentFactory<? extends D> factory;           // Converts a Lucene document to D
  protected final String             defaultFieldName;  // Default search field in the index

  /**
   * Concurrent loading cache to save relevance scores (similarities) for all results for a given
   * Lucene {@link org.apache.lucene.search.Query Query} and
   * Lucene {@link org.apache.lucene.search.similarities.Similarity Similarity}
   * Used in {@link #getSimilarityFiltered(Similarity, Query, Query)}
   */
  private final LoadingCache<Pair<Query, Similarity>, TIntFloatMap> similarityCache =
      CacheBuilder.newBuilder()
          .concurrencyLevel(Runtime.getRuntime().availableProcessors() * 2)
          .expireAfterAccess(5, TimeUnit.MINUTES)
          .build(new CacheLoader<Pair<Query, Similarity>, TIntFloatMap>() {
            @Override
            public TIntFloatMap load(@Nonnull final Pair<Query, Similarity> pair) throws Exception {
              return getSimilarities(pair.second(), pair.first());
            }
          });

  /**
   * Locks used by getSimilarity and friends
   */
  private final ReentrantReadWriteLock rwlock = new ReentrantReadWriteLock();
  private final ReentrantReadWriteLock.ReadLock rlock = rwlock.readLock();
  private final ReentrantReadWriteLock.WriteLock wlock = rwlock.writeLock();


  /**
   * Create a new LuceneSearchEngine
   *
   * @param index        (file) path to Lucene index
   * @param analyzer     analyzer used to analyze text (e.g. EnglishAnalyzer)
   * @param defaultField default search field (e.g. TEXT)
   * @param factory      document factory used to convert a LuceneDocument to type <D>
   */
  public LuceneSearchEngine(final String index,
                            final Analyzer analyzer,
                            final String defaultField,
                            final DocumentFactory<D> factory) {
    assert !Strings.isNullOrEmpty(index) : "Index was null or empty!";
    assert Objects.nonNull(analyzer) : "Analyzer was null!";
    assert !Strings.isNullOrEmpty(defaultField) : "Default field was null or empty!";


    try {
      this.factory = factory;
      this.defaultFieldName = defaultField;
      this.analyzer = analyzer;
      this.parser = new QueryParser(defaultField, analyzer);
      final Directory dir = NIOFSDirectory.open(Paths.get(index));
      this.readerManager = new ReaderManager(dir);
      this.searcherManager = new SearcherManager(dir, new SearcherFactory());
    } catch (IOException ex) {
      throw new RuntimeException("Failed to create Lucene search engine", ex);
    }
  }

  /**
   * Create a new LuceneSearchEngine
   *
   * @param index        (file) path to Lucene index
   * @param analyzer     analyzer used to analyze text (e.g. EnglishAnalyzer)
   * @param defaultField default search field (e.g. TEXT)
   * @param factory      document factory used to convert a LuceneDocument to type <D>
   * @param similarity default similarity
   */
  public LuceneSearchEngine(final String index,
                            final Analyzer analyzer,
                            final String defaultField,
                            final DocumentFactory<? extends D> factory,
                            final Similarity similarity) {
    assert !Strings.isNullOrEmpty(index) : "Index was null or empty!";
    assert Objects.nonNull(analyzer) : "Analyzer was null!";
    assert !Strings.isNullOrEmpty(defaultField) : "Default field was null or empty!";


    try {
      this.factory = factory;
      this.defaultFieldName = defaultField;
      this.analyzer = analyzer;
      this.parser = new QueryParser(defaultField, analyzer);
      final Directory dir = NIOFSDirectory.open(Paths.get(index));
      this.readerManager = new ReaderManager(dir);
      this.searcherManager = new SearcherManager(this.readerManager.acquire(), new SearcherFactory() {
        @Override
        public IndexSearcher newSearcher(IndexReader reader, IndexReader previousReader) throws IOException {
          final IndexSearcher searcher = super.newSearcher(reader, previousReader);
          searcher.setSimilarity(similarity);
          return searcher;
        }
      });
      log.info("Opened new {} from  index at {}", this.getClass().getSimpleName(), index);
    } catch (IOException ex) {
      throw new RuntimeException("Failed to create Lucene search engine", ex);
    }
  }

  /**
   * Create a new LuceneSearchEngine
   *
   * @param dir        (file) path to Lucene index
   * @param analyzer     analyzer used to analyze text (e.g. EnglishAnalyzer)
   * @param defaultField default search field (e.g. TEXT)
   * @param factory      document factory used to convert a LuceneDocument to type <D>
   * @param similarity default similarity
   */
  public LuceneSearchEngine(
      final Directory dir,
      final Analyzer analyzer,
      final String defaultField,
      final DocumentFactory<? extends D> factory,
      final Similarity similarity) {
    assert Objects.nonNull(analyzer) : "Analyzer was null!";
    assert !Strings.isNullOrEmpty(defaultField) : "Default field was null or empty!";


    try {
      this.factory = factory;
      this.defaultFieldName = defaultField;
      this.analyzer = analyzer;
      this.parser = new QueryParser(defaultField, analyzer);
      this.readerManager = new ReaderManager(dir);
      this.searcherManager = new SearcherManager(this.readerManager.acquire(), new SearcherFactory() {
        @Override
        public IndexSearcher newSearcher(IndexReader reader, IndexReader previousReader) throws IOException {
          final IndexSearcher searcher = super.newSearcher(reader, previousReader);
          searcher.setSimilarity(similarity);
          return searcher;
        }
      });
      log.info("Opened new {} from  index at {}", this.getClass().getSimpleName(), dir);
    } catch (IOException ex) {
      throw new RuntimeException("Failed to create Lucene search engine", ex);
    }
  }

  public QueryParser getParser() {
    return parser;
  }

  /**
   * Search for given query String using given Similarity and return up to specified number of
   * results
   *
   * @param sim        Similarity (relevance model) used to score documents
   * @param query      Query String to search
   * @param numResults Maximum number of returned documents
   * @return {@link LuceneSearchResultsList LuceneSearchResultsList} containing search results
   */
  public LuceneSearchResultsList<D> search(final Similarity sim, final String query, int numResults) {
    return search(sim, newParsedQuery(query), numResults);
  }

  /**
   * Parse the given CharSequence using this index's
   * {@link QueryParser QueryParser} to produce a Lucene
   * {@link Query Query}.
   *
   * @param string String to parse
   * @return Parsed Lucene Query
   */
  public Query newParsedQuery(CharSequence string) {
    try {
      return parser.parse(string.toString());
    } catch (ParseException ex) {
      throw new RuntimeException("Failed to parse", ex);
    }
  }

  /**
   * Search for given Lucene {@link Query query} using given Similarity and return up to specified
   * number of results
   *
   * @param sim        Similarity (relevance model) used to score documents
   * @param query      Lucene Query to search
   * @param numResults Maximum number of returned documents
   * @return {@link LuceneSearchResultsList LuceneSearchResultsList} containing search results
   */
  public LuceneSearchResultsList<D> search(final Similarity sim, final Query query, int numResults) {
    return withSearcher(sim, searcher -> {
      try {
        final TopDocs topDocs = searcher.search(query, numResults);
        final ScoreDoc[] scores = topDocs.scoreDocs;

        final int resultSize = Math.min(scores.length, numResults);

        // Create our result list
        //noinspection unchecked
        final LuceneResult<D>[] list = (LuceneResult<D>[]) new LuceneResult[ resultSize ];
        for (int i = 0; i < resultSize; i++) {
          list[i] = new LuceneResult<>(null, scores[i].score, i + 1, scores[i].doc);
        }

        // Make a copy and sort by docId to improve read times
        //noinspection unchecked
        final LuceneResult<D>[] listByDocId = (LuceneResult<D>[]) new LuceneResult[ resultSize ];
        System.arraycopy(list, 0, listByDocId, 0, resultSize);
        Arrays.sort(listByDocId, Comparator.comparingInt(LuceneResult::getLuceneDocId));
        for (int i = 0; i < resultSize; i++) {
          // Modify lucene result in-place
          listByDocId[i].setValue(factory.build(searcher.getIndexReader(), scores[i].doc));
        }

        return new LuceneSearchResultsList<>(topDocs.getMaxScore(), topDocs.totalHits, Arrays.asList(list));
      } catch (IOException ex) {
        throw new RuntimeException("Failed to search", ex);
      }
    });
  }

  /**
   * Search for given Lucene {@link Query query} using given Similarity and return up to specified
   * number of results
   *
   * @param sim        Similarity (relevance model) used to score documents
   * @param query      Lucene Query to search
   * @param docSet     set of Lucene document IDs to score
   * @return {@link Int2FloatMap scoresByLuceneId} map containing scores for each lucene document ID
   */
  public Int2FloatMap getDocSetScores(final Similarity sim, final Query query, BitSet docSet) {
    return withSearcher(sim, searcher -> {
      try {
        final FilteringScoreCollector fsc = new FilteringScoreCollector(docSet);
        searcher.search(query, fsc);
        return fsc.getScores();
      } catch (IOException ioe) {
        throw new RuntimeException(ioe);
      }
    });
  }

  public Int2FloatMap getDocSetScores2(final Similarity sim, final Query query, BitSet docSet) {
    return withSearcher(sim, searcher -> {
      try {
        final Int2FloatMap scores = new Int2FloatOpenHashMap();
        final Weight weight = searcher.createNormalizedWeight(query, true);
        for (LeafReaderContext context : searcher.getTopReaderContext().leaves()) {
          final Scorer scorer = weight.scorer(context);
          final int maxDoc = context.reader().maxDoc();
          if (scorer == null) {
            log.trace("scorer was null for context: {} maxDoc={}, numDocs={}; query={}", context, maxDoc, context.reader().numDocs(), query);
            continue;
          }
          final DocIdSetIterator it = scorer.iterator();
          assert it != null : "Scorer's DocIdSetIterator was null!";

          // for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i+1)) {
          for (int doc = docSet.nextSetBit(context.docBase); doc >= 0; doc = docSet.nextSetBit(doc + 1)) {
            if (doc == Integer.MAX_VALUE) {
                break; // or (doc + 1) would overflow
            }
            int contextDoc = doc - context.docBase;
            if (contextDoc > maxDoc) {
              break;
            }

            if (scorer.docID() < contextDoc) {
              int scoredId = it.advance(contextDoc);
//              log.trace("Scorer doc ID = {} (iterator doc id = {}; context doc id = {}; global doc id= {})", scorer.docID(), scoredId, contextDoc, doc);
              if (scoredId == DocIdSetIterator.NO_MORE_DOCS) {
                log.trace("Exhausted scorer!");
                break;
              }
              if (scoredId == contextDoc) {
                float score = scorer.score();
                scores.put(doc, score);
              }
            }
          }
        }
        int[] misses = docSet.stream().filter(i -> !scores.containsKey(i)).toArray();
        log.trace("Unable to find {} of {} docs for query {}", misses.length, docSet.cardinality(), query);

//        if (scores.isEmpty()) {
//          throw new IllegalStateException("Unable to find docs " + Arrays.toString(+ " in any IndexReader leaf!");
//        }
        return scores;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }

    });
  }

  public float getDocScore(final Similarity sim, final Query query, int doc) {
    return withSearcher(sim, searcher -> {
      try {
        final Weight weight = searcher.createNormalizedWeight(query, true);
        for (LeafReaderContext context : searcher.getTopReaderContext().leaves()) {
          if (context.docBase > doc) {
            continue;
          }
          int contextDoc = doc - context.docBase;
          if (contextDoc >= context.reader().maxDoc()) {
            continue;
          }

          final Scorer scorer = weight.scorer(context);
          scorer.iterator().advance(contextDoc);
          if (scorer.docID() == contextDoc) {
            return scorer.score();
          }
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      throw new IllegalStateException("Unable to find doc" + doc + " in any IndexReader leaf!");
    });
  }

  /**
   * Acquires an IndexSearcher using the given Similarity and applies the given function
   *
   * @param sim  Similarity (relevance model) used by IndexSearcher
   * @param func Function to apply
   * @param <X>  Return type of func
   * @return result of func(indexSearcher)
   */
  protected <X> X withSearcher(Similarity sim, Function<? super IndexSearcher, X> func) {
    try {
      final IndexSearcher searcher = searcherManager.acquire();
      searcher.setSimilarity(sim);
      try {
        return func.apply(searcher);
      } finally {
        searcherManager.release(searcher);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Search for given query String using default Similarity and return up to specified number of
   * results
   *
   * @param query      Query String to search
   * @param numResults Maximum number of returned documents
   * @return {@link LuceneSearchResultsList LuceneSearchResultsList} containing search results
   */
  public LuceneSearchResultsList<D> search(final String query, int numResults) {
    return search(newParsedQuery(query), numResults);
  }

  /**
   * Search for given Lucene {@link Query query} using default Similarity and return up to specified
   * number of results
   *
   * @param query      Lucene Query to search
   * @param numResults Maximum number of returned documents
   * @return {@link LuceneSearchResultsList LuceneSearchResultsList} containing search results
   */
  @Override
  public LuceneSearchResultsList<D> search(final Query query, int numResults) {
    return search(Similarities.BM25.similarity, query, numResults);
  }

  /**
   * Token given CharSequence using this index's analyzer as if it were in the default field
   *
   * @param string CharSequence to tokenize
   * @return List of tokens
   */
  public List<String> tokenize(CharSequence string) {
    return LuceneUtils.tokenize(string, defaultFieldName, analyzer);
  }

  /**
   * Get total number of hits for a query in the index
   *
   * @param query Query to search for
   * @return number of hits
   */
  public int getHitCount(Query query) {
    return withSearcher(searcher -> {
      try {
        final TotalHitCountCollector collector = new TotalHitCountCollector();
        searcher.search(query, collector);
        return collector.getTotalHits();
      } catch (IOException ex) {
        throw new RuntimeException(ex);
      }
    });
  }

  /**
   * Acquires an IndexSearcher with the default Similarity and applies the given function
   *
   * @param func Function to apply
   * @param <X>  Return type of func
   * @return result of func(indexSearcher)
   */
  public <X> X withSearcher(Function<? super IndexSearcher, X> func) {
    return withSearcher(Similarities.BM25.similarity, func);
  }

  /**
   * Get total number of documents in the index
   *
   * @return number of documents
   */
  public long getNumberOfDocuments() {
    return withReader(dr -> (long) dr.numDocs());
  }

  /**
   * Acquires a DirectoryReader and applies the given function
   *
   * @param func Function to Apply
   * @param <X>  Return type of func
   * @return Result of func(directoryReader)
   */
  public <X> X withReader(Function<? super DirectoryReader, X> func) {
    try {
      DirectoryReader reader = readerManager.acquire();
      try {
        return func.apply(reader);
      } finally {
        readerManager.release(reader);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Tokenize the given String to produce a new PhraseQuery using this index's analyzer
   *
   * @param field  Field assumed by analyzer
   * @param string CharSequence to be tokenized
   * @return Resultant PhraseQuery
   */
  public PhraseQuery newPhraseQuery(final String field, final CharSequence string) {
    final PhraseQuery.Builder query = new PhraseQuery.Builder();
    for (final String token : tokenize(field, string)) {
      query.add(new Term(field, token));
    }
    return query.build();
  }

  /**
   * Token given CharSequence using this index's analyzer as if it were in the given field
   *
   * @param field  Name of field assumed by analyzer
   * @param string CharSequence to tokenize
   * @return List of tokens
   */
  public List<String> tokenize(String field, CharSequence string) {
    return LuceneUtils.tokenize(string, field, analyzer);
  }

  /**
   * Close this index and release all resources
   */
  @Override
  public void close() {
    try {
      searcherManager.close();
      readerManager.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    analyzer.close();
  }

  /**
   * Get the length of a given field for a given Lucene document
   *
   * @param luceneId the ID of a document as assigned by Lucene
   * @param field    the desired field
   * @return number of terms in the given field for the specified document
   */
  public long getFieldLength(final int luceneId, String field) {
    return withSearcher(searcher -> {
      try {
        final Terms termVector = searcher.getIndexReader().getTermVector(luceneId, field);
        try {
          return termVector.size();
        } catch (NullPointerException e) {
          log.debug("No term vector for {} in document {}", field, luceneId);
          return 0L;
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Get the IDF, or inverse document frequency
   *
   * @param query Query to search
   * @return "log(totalDocuments / (DF + 1))"
   */
  public double getInverseDocumentFrequency(Query query) {
    return withReader(reader -> {
      final int total = reader.numDocs();
      return Math.log(total / (getDocumentFrequency(query) + 1));
    });
  }

  /**
   * Get the DF, or number of documents that match a query
   *
   * @param query Query to search
   * @return total number of hits for given query
   */
  public double getDocumentFrequency(Query query) {
    final TotalHitCountCollector collector = new TotalHitCountCollector();
    return withSearcher(searcher -> {
      try {
        searcher.search(query, collector);
        return (double) collector.getTotalHits();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Get the normalized IDF
   *
   * @param query Query to search
   * @return "log(1 + totalDocuments / (DF + 1))"
   */
  public double getNormalizedInverseDocumentFrequency(Query query) {
    return withReader(reader -> {
      final int total = reader.numDocs();
      return Math.log(1 + total / (getDocumentFrequency(query) + 1));
    });

  }

  /**
   * Get the normalizeD TF, or term frequency, in a given document and field
   *
   * @param query Query containing span of terms
   * @param id    lucene Document id
   * @param field field
   * @return "0.5 + (0.5 * TF / maxTF)"
   */
  public double getNormalizedTermFrequency(SpanQuery query, int id, String field) {
    SpanWeight spanWeight = withSearcher(Unsafe.function(searcher -> query.createWeight(searcher, false)));

    return withReader(reader -> {
      BytesRef bytes;
      Terms terms;
      Spans spans;
      TermsEnum termsEnum;

      long tf = 0, maxTF = 0;
      int docId, maxDoc;

      for (final LeafReaderContext leaf : reader.leaves()) {
        docId = id - leaf.docBase;
        try (final LeafReader leafReader = leaf.reader()) {
          terms = leafReader.getTermVector(docId, field);
          if (terms == null) {
            log.warn("No terms found for document {} in field {}", id, field);
            return 0.5;
          }
          termsEnum = terms.iterator();
          bytes = termsEnum.next();
          while (bytes != null) {
            tf = termsEnum.totalTermFreq();
            if (tf > maxTF) {
              maxTF = tf;
            }
            bytes = termsEnum.next();
          }
          tf = 0;
          spans = spanWeight.getSpans(leaf, SpanWeight.Postings.POSITIONS);
          while (spans.nextStartPosition() != Spans.NO_MORE_POSITIONS) {
            tf += 1;
          }
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }

      return 0.5 + (0.5 * tf / ((double) maxTF));
    });
  }

  public double getSimilarity(final Similarity similarity,
                              final Query query,
                              final int docId) {
    return getSimilarity(similarity, query, Integer.toString(query.hashCode()), docId);
  }


  public double getSimilarity(final Similarity similarity,
                              final Query query,
                              final String name,
                              final int docId) {
    final Pair<Query, Similarity> key = Pair.of(query, similarity);
    rlock.lock();
    TIntFloatMap result;
    try {
      result = similarityCache.getIfPresent(key);
    } finally {
      rlock.unlock();
    }

    if (result == null) {
      wlock.lock();
      try {
        result = similarityCache.getUnchecked(key);
      } finally {
        wlock.unlock();
      }
    }
    return result.get(docId);
  }

  /**
   * Get the similarity of a document to a query using the given filter
   * It is assumed that the filter restricts the results to a single document
   *
   * @param similarity Similarity metric
   * @param query      Query to search
   * @param filter     CloseablePredicate used to restrict results to single document
   * @return score of filtered document to given query
   */
  private double getSimilarityFiltered(final Similarity similarity,
                                       final Query query,
                                       final Query filter) {
    return withSearcher(similarity, searcher -> {
      try {
        final BooleanQuery.Builder filtered = new BooleanQuery.Builder();
        filtered.add(query, BooleanClause.Occur.MUST);
        filtered.add(filter, BooleanClause.Occur.FILTER);
        return (double) searcher.search(filtered.build(), 1).scoreDocs[0].score;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Gets the binary hits for a given query (i.e., without scores) as a bitset
   * where each bit corresponds to a given document's lucene ID
   *
   * @param query query to search
   * @return resultant bitset
   */
  public BitSet getHits(final Query query) {
    return withSearcher(s -> {
      try {
        return s.search(query, new CollectorManager<BitsetCollector, BitSet>() {
          @Override
          public BitsetCollector newCollector() throws IOException {
            return new BitsetCollector();
          }

          @Override
          public BitSet reduce(Collection<BitsetCollector> collectors) throws IOException {
            final BitSet result = new BitSet(withReader(DirectoryReader::numDocs));
            for (BitsetCollector c : collectors) {
              result.or(c.hits);
            }
            return result;
          }
        });
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Produces a map of lucene IDs to similarity scores for a given query
   *
   * @param sim   Similarity measure
   * @param query Query to search
   * @return Map of lucene ID -> similarity score
   */
  public TIntFloatMap getSimilarities(final Similarity sim, final Query query) {
    return withSearcher(sim, searcher -> {
      final AllDocsCollector collector = AllDocsCollector.create();
      try {
        searcher.search(query, collector);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      return collector.getScores();
    });
  }

  /**
   * Get the number of unique terms in the index across all fields
   *
   * @return vocabulary size
   */
  public long getVocabularySize() {
    return withReader(LuceneUtils::getVocabularySize);
  }

  public Query newSpanQuery(CharSequence string, String field) {
    return LuceneUtils.getSpanQuery(string, field, analyzer);
  }

  public Query newSpanQuery(CharSequence string, String field, double weight) {
    return LuceneUtils.getSpanQuery(string, field, analyzer, 1.0);
  }

  public Query newSpanQueryWithSlop(CharSequence string, String field, int slop) {
    return LuceneUtils.getSpanQuery(string, field, analyzer, 1.0);
  }

  public Query newSpanQuery(CharSequence string, String field, Analyzer analyzer, double weight) {
    return LuceneUtils.getSpanQuery(string, field, analyzer, 1.0);
  }

  public <T extends CharSequence, K extends Weighted<T>> BooleanQuery newBooleanQuery(Iterable<K> it) {
    return newBooleanQuery(it, defaultFieldName, BooleanClause.Occur.SHOULD);
  }

  public <T extends CharSequence, K extends Weighted<T>> BooleanQuery newBooleanQuery(Iterable<K> it, String field) {
    return newBooleanQuery(it, field, BooleanClause.Occur.SHOULD);
  }

  public <T extends CharSequence, K extends Weighted<T>> BooleanQuery newBooleanQuery(Iterable<K> it, String field, BooleanClause.Occur occur) {
    return LuceneUtils.getBooleanQuery(it, field, analyzer, occur);
  }

  public Analyzer getAnalyzer() {
    return analyzer;
  }

  /**
   * Collects hits for a given query into a bitset
   */
  private class BitsetCollector implements Collector {
    final BitSet hits = new BitSet(withReader(DirectoryReader::numDocs));


    public LeafCollector getLeafCollector(LeafReaderContext context) throws IOException {
      return new LeafCollector() {
        final int docBase = context.docBase;

        @Override
        public void setScorer(Scorer scorer) throws IOException {
          // do nothing
        }

        @Override
        public void collect(int doc) throws IOException {
          hits.set(doc);
        }
      };
    }

    /**
     * Indicates if document scores are needed by this collector.
     *
     * @return {@code true} if scores are needed.
     */
    @Override
    public boolean needsScores() {
      return false;
    }
  }
}
