package edu.utdallas.hltri.inquire.scripts

import java.io._
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{ConcurrentSkipListSet, ExecutorService, ForkJoinPool, TimeUnit}
import javax.annotation.concurrent.ThreadSafe

import com.google.common.base.Strings
import edu.utdallas.hltri.io.IOUtils
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.document._
import org.apache.lucene.index.IndexWriterConfig.OpenMode
import org.apache.lucene.index.{IndexOptions, IndexWriter, IndexWriterConfig}
import org.apache.lucene.search.similarities.BM25Similarity
import org.apache.lucene.store.MMapDirectory
import org.jdom2.filter.Filters
import org.jdom2.input.SAXBuilder
import org.jdom2.xpath.{XPathExpression, XPathFactory}
import org.jdom2.{Element, JDOMException, Document => XMLDocument}

import scala.collection.JavaConverters._


/**
 * Created by travis on 7/17/14.
 */
class PMCIndexer(
  private val indexFile: File
  ) extends Closeable with Logging {

  private val analyzer: Analyzer = new EnglishAnalyzer()

  // Open Lucene index for writing (thread-safe)
  private val writer: IndexWriter = {
    val directory = new MMapDirectory(indexFile.toPath)
    val config = new IndexWriterConfig(analyzer).setOpenMode(OpenMode.CREATE_OR_APPEND).setSimilarity(new BM25Similarity(1.2f, 0.25f))
    new IndexWriter(directory, config)
  }
  log.debug(s"Opened index at $indexFile for writing")

  // Custom text field that stores **everything** and analyzes
  private val textField: FieldType = {
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

  // Thread-local JDOM parser with all DTD validation disabled
  private val builder: ThreadLocal[SAXBuilder] = Local{
    val self = new SAXBuilder()
    self.setFeature("http://xml.org/sax/features/validation", false)
    self.setFeature("http://apache.org/xml/features/nonvalidating/load-dtd-grammar", false)
    self.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false)
    self
  }

  // Thread-safe X-Path factory
  private val xPath: XPathFactory = XPathFactory.instance()

  /* X-Path expressions courtesy of JDOM
   * Set as thread-local because they are not thread-safe
   * Set to only return the text, rather than the matching element
   */
  private val xBody = Local(xPath.compile("/article/body", Filters.element()))
  private val xAbstract = Local(xPath.compile("/article/front/article-meta/abstract", Filters.element()))
  private val xArticleTitle = Local(xPath.compile("/article/front/article-meta/title-group/article-title", Filters.element()))
  private val xSections = Local(xPath.compile("/article/body/sec", Filters.element()))
  private val xJournalTitle = Local(xPath.compile("/article/front/journal-meta/journal-title-group/journal-title", Filters.element()))
  private val xPMC = Local(xPath.compile("/article/front/article-meta/article-id[@pub-id-type='pmc']", Filters.element()))
  private val xDOI = Local(
    xPath.compile("/article/front/article-meta/article-id[@pub-id-type='doi']", Filters.element())
  )
  private val xKW = Local(
    xPath.compile("/article/front/article-meta/kwd-group/kwd", Filters.element())
  )
  private val xPMID = Local(
    xPath.compile("/article/front/article-meta/article-id[@pub-id-type='pmid']", Filters.element())
  )

  // Re-useable StringBuilder to maximize those efficiency gainz (thread-local)
  private val stringBuilder = Local(new StringBuilder())

  private val totalStart = System.nanoTime()

  // Stores how many documents we've index thus far
  private val count = new AtomicInteger(0)

  private class EmptyFieldException(msg: String) extends RuntimeException(msg)

  /* Convenience method for constructing a Lucene Field object with the given name and type by:
   * (1) parsing the implicitly passed XMLDocument using the given xPath expression
   * (2) concatenating all matching texts after trimming excess whitespace
   */
  private def addField(name: String, xpath: XPathExpression[Element], fieldType: FieldType, required: Boolean = true)
    (implicit document: Document, xml: XMLDocument) {
    for (text: Element <- xpath.evaluate(xml).asScala) {
      val string = text.getValue
      if (Strings.isNullOrEmpty(string) && required) {
        throw new EmptyFieldException(s"empty ${xpath.getExpression}")
      }
      document.add(new Field(name, string, fieldType))
    }
  }

  private def addMergedField(name: String, xpath: XPathExpression[Element], fieldType: FieldType, required: Boolean = true)
    (implicit document: Document, xml: XMLDocument) {
    val sb = stringBuilder.get
    sb.setLength(0)
    for (text: Element <- xpath.evaluate(xml).asScala) {
      if (sb.nonEmpty) {sb += '\n'}
      sb ++= text.getValue
    }
    if (sb.isEmpty && required) {
      throw new EmptyFieldException(s"empty ${xpath.getExpression}")
    }
    document.add(new Field(name, sb.toString(), fieldType))
  }

  /**
    * Index the given file (thread-safe)
    * @param file: file to index
    */
  @ThreadSafe
  def indexFile(file: File): Unit = {
    val start = System.nanoTime() // Save start time

    // Parse XML and let it be implicitly passed to the field(...) method
    implicit val xml: XMLDocument = builder.get.build(file)

    // Create an empty lucene document & populate with NXML-fields
    implicit val document = new Document()
    try {
      addField("pmc_id", xPMC.get, StringField.TYPE_STORED)
      addField("pm_id", xPMID.get, StringField.TYPE_STORED, required = false)
      addField("doi", xDOI.get, StringField.TYPE_STORED, required = false)
      addField("journal_title", xJournalTitle.get, textField, required = false)
      addField("article_title", xArticleTitle.get, textField)
      addField("abstract", xAbstract.get, textField, required = false)
      addField("body", xBody.get, textField, required = false)
      addField("keywords", xKW.get, textField, required = false)

      addMergedField("text", xSections.get, textField, required = false)

      // Manually construct additional fields
      document.add(new Field("filename", file.getName, StringField.TYPE_STORED))
      document.add(new Field("path", file.getAbsolutePath, StringField.TYPE_STORED))

      // Write document to index, log it, and synchronize the index every 100 documents
      writer.addDocument(document)
      log.info(f"Indexed document ${count.incrementAndGet()}%,d (${file.getName}%s) in ${(System.nanoTime() - start).toDouble / 1e9}%5.3f s")
    } catch {
      case ex: EmptyFieldException =>
        log.info(
          f"Skipped document ${count.incrementAndGet()}%,d ($file%s) after ${
            (System.nanoTime() - start).toDouble / 1e9
          }%5.3f s because ${ex.getMessage}"
        )
    } finally {
      if (count.get() % 1000 == 0) {
        sync()
      }
    }
  }

  /**
    * Saves the current index to the disk
    */
  @ThreadSafe
  def sync(): Unit = {
    synchronized {
      log.debug(s"Synchronizing index...")
      writer.commit()
    }
  }

  /**
    * Saves the index to the disk and closes it, releasing all resources
    */
  override def close(): Unit = {
    synchronized {
      writer.commit()
      writer.close()
      log.info(f"Finished indexing ${count.get()}%,d documents in ${(System.nanoTime() - totalStart).toDouble / 3.6e12}%5.3f hours")
    }
  }
}

/**
  * Script for creating a new PubMed Central index
  * Usage: PMCIndexer /path/to/input/directory /path/to/index
  *
  * Also writes /path/to/input/directory.failed containing the names of files that were unable to be parsed or read
  */
object PMCIndexer extends App with Logging {
  private final val input = new File(args(0))
  private final val index = new File(args(1))
  private final val failedFile = new File(index.getAbsolutePath + ".failed")
  private final val service: ExecutorService = new ForkJoinPool()
  private final val failed = new ConcurrentSkipListSet[File]().asScala

  using(new PMCIndexer(index)) { indexer =>
    for (file <- IOUtils.`lazy`.iterateWithSuffix(input, ".nxml").asScala) {
      service.submit(
        new Runnable() {
          override def run(): Unit = {
            try {
              indexer.indexFile(file)
            } catch {
              case ex: JDOMException => log.error(s"Malformed XML in $file: ${ex.getMessage}")
                failed += file
              case ex: IOException => log.error(s"Failed to read $file: ${ex.getMessage}")
                failed += file
              case ex: Throwable => throw new RuntimeException(ex)
            }
          }
        }
      )
    }
    service.shutdown()
    service.awaitTermination(2L, TimeUnit.DAYS)
  }

  log.info(f"${failed.size}%,d documents failed to index.")
  log.info(f"Saving list of failed documents to $failedFile...")
  if (failed.nonEmpty) {
    using(new BufferedWriter(new FileWriter(failedFile))) { writer =>
      for (fail <- failed) {
        writer.write(input.toPath.relativize(fail.toPath).toString)
        writer.newLine()
      }
    }
  }
  log.info("Complete!")
}
