package edu.utdallas.hltri.inquire.lucene;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import edu.utdallas.hltri.logging.Logger;

@SuppressWarnings({"unused", "WeakerAccess"})
public final class IndexMerger {
  private static final Logger log = Logger.get(IndexMerger.class);
  private final List<Directory> indexShards = new ArrayList<>();

  private IndexMerger() {
  }

  private IndexMerger(Collection<Directory> indexShards) {
    this.indexShards.addAll(indexShards);
  }

  @SuppressWarnings("UnusedReturnValue")
  public IndexMerger addIndexShard(Path path) {
    try {
      return addIndexShard(new NIOFSDirectory(path));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public IndexMerger addIndexShard(Directory directory) {
    indexShards.add(directory);
    return this;
  }

  public static IndexMerger forShards(Path... paths) {
    return forShards(Arrays.asList(paths));
  }

  public static IndexMerger forShards(Iterable<Path> paths) {
    final IndexMerger merger = new IndexMerger();
    for (final Path path : paths) {
      merger.addIndexShard(path);
    }
    return merger;
  }

  public void mergeTo(Path path, Analyzer analyzer) {
    try {
      final Directory indexDir = new NIOFSDirectory(path);
      final IndexWriterConfig config = new IndexWriterConfig(analyzer)
          .setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
          .setSimilarity(new BM25Similarity());
      final IndexWriter indexWriter = new IndexWriter(indexDir, config);
      log.info("Creating merged index at {} with analyzer {}...", path, analyzer);
      indexWriter.addIndexes(indexShards.toArray(new Directory[ indexShards.size() ]));
      log.info("Committing changes...");
      indexWriter.commit();
//      log.info("Optimizing index...");
//      indexWriter.forceMerge(8);
      log.info("Closing index...");
      indexWriter.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
