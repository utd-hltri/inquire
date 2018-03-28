package edu.utdallas.hltri.inquire.scripts;

import edu.utdallas.hltri.framework.Commands;
import edu.utdallas.hltri.framework.ProgressLogger;
import edu.utdallas.hltri.logging.Logger;
import it.unimi.dsi.fastutil.objects.Object2IntArrayMap;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import picocli.CommandLine.Parameters;

public class LuceneIndexDeduplicator implements Runnable {
  private static final Logger log = Logger.get(LuceneIndexDeduplicator.class);

  @Parameters(index = "0", paramLabel = "INDEX-PATH")
  private Path indexPath;

  @Parameters(index = "1", paramLabel = "UNIQUE-FIELD")
  private String uniqueField;

  private static class ReaderId {
    final IndexReader reader;
    final int id;

    public ReaderId(IndexReader reader, int id) {
      this.reader = reader;
      this.id = id;
    }
  }

  @Override
  public void run() {
    final Map<String, ReaderId> docsByPmids = new HashMap<>();
    try(final ProgressLogger plog = ProgressLogger.indeterminateSize("de-duplicating", 1, TimeUnit.MINUTES);
        final Directory indexDir = new NIOFSDirectory(indexPath);
        final Analyzer analyzer = new EnglishAnalyzer()) {
      final IndexWriterConfig config = new IndexWriterConfig(analyzer)
          .setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
          .setSimilarity(new BM25Similarity());
      try (IndexWriter indexWriter = new IndexWriter(indexDir, config)) {
        try (IndexReader reader = DirectoryReader.open(indexWriter)) {
          LeafReader leafReader;
          Terms leafTerms;
          for (LeafReaderContext lrc : reader.leaves()) {
            leafReader = lrc.reader();
            leafTerms  = leafReader.terms(uniqueField);
            if (leafTerms == null) {
              continue;
            }
            final TermsEnum terms = leafTerms.iterator();
            PostingsEnum postings = null;
            for (BytesRef bytes = null; ((bytes = terms.next()) != null); ) {
              final Bits liveDocs = leafReader.getLiveDocs();
              final String pmid = terms.term().utf8ToString();
//              log.debug("Found PMID {}", pmid);
              final boolean haveSeenPmid = docsByPmids.containsKey(pmid);
              postings = terms.postings(postings, PostingsEnum.NONE);
              int lastDoc = -1;
              for (int docId; ((docId = postings.nextDoc()) != PostingsEnum.NO_MORE_DOCS); ) {
                // If doc is marked as deleted we should skip it
                if (liveDocs != null && !liveDocs.get(docId)) {
                  continue;
                }
                if (haveSeenPmid) {
                  ReaderId prev = docsByPmids.get(pmid);
                  indexWriter.tryDeleteDocument(prev.reader, prev.id);
                  plog.update("deleting {}", prev.id);
                } else {
                  if (lastDoc != -1) {
                    indexWriter.tryDeleteDocument(leafReader, lastDoc);
                    plog.update("deleting {}", lastDoc);
                  }
                  lastDoc = docId;
                }
                docsByPmids.put(pmid, new ReaderId(leafReader, docId));
              }
            }
          }
          indexWriter.flush();
          log.info("{} pending deletions", reader.numDeletedDocs());
        }
        indexWriter.maybeMerge();
        indexWriter.commit();
        indexWriter.maybeMerge();
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static void main(String... args) {
    Commands.run(new LuceneIndexDeduplicator(), args);
  }
}
