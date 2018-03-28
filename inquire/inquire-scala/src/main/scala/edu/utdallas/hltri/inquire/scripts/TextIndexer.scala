package edu.utdallas.hltri.inquire.scripts

import java.io._
import java.nio.file._
import java.nio.file.attribute.BasicFileAttributes
import java.util.concurrent._
import javax.annotation.concurrent.ThreadSafe

import com.google.common.base.Charsets
import edu.utdallas.hltri.framework.ProgressLogger
import edu.utdallas.hltri.io.IOUtils
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.document._
import org.apache.lucene.index.IndexWriterConfig.OpenMode
import org.apache.lucene.index._
import org.apache.lucene.search.similarities.BM25Similarity
import org.apache.lucene.store.NIOFSDirectory
import org.jdom2.{Document => XMLDocument}

/**
 * Created by travis on 7/17/14.
 */
class TextIndexer(private[this] val indexPath: Path) extends Closeable with Logging {

  // Open Lucene index for writing (thread-safe)
  private[this] val writer = {
    val directory = new NIOFSDirectory(indexPath)
    val config = new IndexWriterConfig(TextIndexer.analyzer)
                 .setOpenMode(OpenMode.CREATE_OR_APPEND)
                 .setSimilarity(new BM25Similarity())
    new IndexWriter(directory, config)
  }

  log.debug(s"Opened index at $indexPath for writing")

  // Custom text field that stores **everything** and analyzes
  val textField: FieldType = {
    val self = new FieldType(TextField.TYPE_STORED)
    self.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)
    self.setTokenized(true)
    self.setStored(true)
    self.setStoreTermVectors(true)
    self.setStoreTermVectorPositions(true)
    self.setStoreTermVectorOffsets(true)
    self.setStoreTermVectorPayloads(true)
    self.freeze()
    self
  }

  val plog: ProgressLogger = ProgressLogger.indeterminateSize("indexing", 10, TimeUnit.SECONDS)

  val identifier = new Term("id")

  /**
   * Index the given file (thread-safe)
   * @param file: file to index
   */
  @ThreadSafe
  def indexFile(file: Path): Unit = {
    val document = new Document()
    val filename = file.getFileName.toString
    val id = IOUtils.removeExtension(filename)
    document.add(new Field("text", com.google.common.io.Files.toString(file.toFile, Charsets.UTF_8), textField))
    document.add(new Field("filename", filename, StringField.TYPE_STORED))
    document.add(new Field("path", file.toAbsolutePath.toString, StringField.TYPE_STORED))
    document.add(new Field("id", id, StringField.TYPE_STORED))
    writer.updateDocument(new Term("id", id), document)
    plog.update("indexed {}", id)
  }

  /**
   * Saves the current index to the disk
   */
  @ThreadSafe
  def sync(): Unit = {
    log.debug(s"Synchronizing index...")
    writer.commit()
  }

  /**
   * Saves the index to the disk and closes it, releasing all resources
   */
  override def close(): Unit = {
    synchronized {
      writer.commit()
      writer.close()
      plog.close()
    }
  }
}

object TextIndexer extends App {
  val analyzer: Analyzer = new EnglishAnalyzer(/*{
    new CharArraySet(java.util.Arrays
      .asList("a",
        "an",
        "that",
        "the",
        "their",
        "then",
        "this",
        "which"), false)
  }*/)

  val text = Paths.get(args(0))
  val index = Paths.get(args(1))

  private[this] val service = new ForkJoinPool()

  val semaphore = new Semaphore(64)

  for (indexer <- managed(new TextIndexer(index))) {
    Files.walkFileTree(text, new SimpleFileVisitor[Path]() {
      override def visitFile(file: Path, attrs: BasicFileAttributes): FileVisitResult = {
        if (attrs.isRegularFile) {
          semaphore.acquire()
          service.execute(new Runnable() {
            override def run(): Unit = {
              indexer.indexFile(file)
              semaphore.release()
            }
          })
        }
        FileVisitResult.CONTINUE
      }
    })
  }

  service.shutdown()
  service.awaitTermination(1, TimeUnit.DAYS)
}
