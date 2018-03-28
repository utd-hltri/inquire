package edu.utdallas.hltri.inquire.scripts

import java.io._
import java.util.concurrent.atomic.AtomicInteger
import javax.annotation.concurrent.ThreadSafe

import com.google.common.base.Strings
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.document._
import org.apache.lucene.index.IndexWriterConfig.OpenMode
import org.apache.lucene.index.{IndexOptions, IndexWriter, IndexWriterConfig}
import org.apache.lucene.search.similarities.BM25Similarity
import org.apache.lucene.store.NIOFSDirectory
import org.apache.lucene.util.Version
import org.jdom2.filter.Filters
import org.jdom2.input.SAXBuilder
import org.jdom2.xpath.{XPathExpression, XPathFactory}
import org.jdom2.{Element, Document => XMLDocument}

import scala.collection.JavaConverters._


/**
 * Created by travis on 7/17/14.
 */
class XMLIndexer(
  private val indexFile: File,
  private val version: Version = Version.LUCENE_5_2_1) extends Closeable with Logging {

  private val analyzer: Analyzer = new EnglishAnalyzer()

  // Open Lucene index for writing (thread-safe)
  private val writer = {
    val directory = new NIOFSDirectory(indexFile.toPath)
    val config = new IndexWriterConfig(analyzer)
                 .setOpenMode(OpenMode.CREATE)
                 .setSimilarity(new BM25Similarity())
    new IndexWriter(directory, config)
  }
  log.debug(s"Opened index at $indexFile for writing")

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

  // Thread-local JDOM parser with all DTD validation disabled
  private val builder = Local {
    val self = new SAXBuilder()
    self.setFeature("http://xml.org/sax/features/validation", false)
    self.setFeature("http://apache.org/xml/features/nonvalidating/load-dtd-grammar", false)
    self.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false)
    self
  }

  // Thread-safe X-Path factory
  private val xPath = XPathFactory.instance()

  // Re-useable StringBuilder to maximize those efficiency gainz (thread-local)
  private val stringBuilder = Local(new StringBuilder())

  private val totalStart = System.nanoTime()

  // Stores how many documents we've index thus far
  private val count = new AtomicInteger(0)

  class EmptyFieldException(msg: String) extends RuntimeException(msg)

  /* Convenience method for constructing a Lucene Field object with the given name and type by:
   * (1) parsing the implicitly passed XMLDocument using the given xPath expression
   * (2) concatenating all matching texts after trimming excess whitespace
   */
  def addField(field: FieldBase)(implicit document: Document, xml: Object): Unit = field match {
    case xPathField: XPathField => {
      for (text: Element <- xPathField.xPathExpr.evaluate(xml).asScala) {
        val string = text.getValue
        if (Strings.isNullOrEmpty(string) && xPathField.required) {
          throw new EmptyFieldException(s"empty ${xPathField.xPathExpr.getExpression}")
        }
        document.add(new Field(xPathField.name, string, xPathField.fieldType))
      }
    }
    case stringField: StringField => {
      document.add(new Field(stringField.name, stringField.value, stringField.fieldType))
    }
  }

  sealed trait FieldBase

  case class XPathField(
    name: String,
    xPathString: String,
    fieldType: FieldType,
    required: Boolean = true) extends FieldBase {

    private val _xPathExpr = Local(xPath.compile(xPathString, Filters.element()))

    def xPathExpr: XPathExpression[Element] = _xPathExpr.get()
  }

  case class StringField(
    name: String,
    value: String,
    fieldType: FieldType = org.apache.lucene.document.StringField.TYPE_STORED
    ) extends FieldBase

  /**
   * Index the given file (thread-safe)
   * @param fields: file to index
   */
  @ThreadSafe
  def indexFile(fields: FieldBase*)(implicit context: Object): Document = {
    val start = System.nanoTime() // Save start time
    // Create an empty lucene document & populate with NXML-fields
    implicit val document = new Document()
    for (field <- fields) {
      addField(field)(document, context)
    }

    // Write document to index, log it, and synchronize the index every 100 documents
    writer.addDocument(document)
    document
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
      log.info(
        f"Finished indexing ${count.get()}%,d documents in ${
          (System.nanoTime() - totalStart).toDouble / 3.6e12
        }%5.3f hours"
      )
    }
  }
}
