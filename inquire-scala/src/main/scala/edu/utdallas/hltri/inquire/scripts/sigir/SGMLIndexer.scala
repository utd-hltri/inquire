package edu.utdallas.hltri.inquire.scripts.sigir

import java.io._
import java.util.Collections
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import javax.xml.stream.XMLInputFactory

import com.google.common.base.Throwables
import edu.utdallas.hltri.concurrent.BoundedThreadPool
import edu.utdallas.hltri.framework.ProgressLogger
import edu.utdallas.hltri.inquire.documents.NewsArticle
import edu.utdallas.hltri.io.IOUtils
import edu.utdallas.hltri.scribe.io.{Corpus, JsonCorpus}
import edu.utdallas.hltri.scribe.text.documents.HasTitle
import edu.utdallas.hltri.scribe.text.{BaseDocument, Document}
import org.apache.commons.compress.compressors.xz.XZCompressorInputStream

import scala.collection.JavaConverters._
/**
 * Created by some wack guy on 10/2/15.
 */
object SGMLIndexer extends App with Logging {
  val inputPaths = args.take(args.length - 1)
  val corpusPath = args(args.length - 1)

  val corpus: Corpus[NewsArticle] = JsonCorpus.at(corpusPath).tiered().build()

  val corpusService = BoundedThreadPool.create(32)

  implicit def runnable(f: => Any): Runnable = new Runnable() { def run() = f }

  val xmlFactory = XMLInputFactory.newInstance()

  val files = inputPaths.flatMap(inputPath => IOUtils.eager.iterateWithSuffix(inputPath, ".xz").asScala)
  for (plog <- managed(ProgressLogger.fixedSize("loading", files.length, 1l, TimeUnit.MINUTES))) {
    for (file <- files) {
      for (reader <- managed(new BufferedInputStream(new SequenceInputStream(Collections
        .enumeration(java
        .util
        .Arrays
        .asList(new ByteArrayInputStream("<DOCS>".getBytes),
          new XZCompressorInputStream(new FileInputStream(file)),
          new ByteArrayInputStream("</DOCS>".getBytes)))))); xml <- managed(xmlFactory.createXMLEventReader(reader, "Cp1252"))) {

        var firstHead = true
        var firstBy = true
        var firstDate = true
        var firstText = true

        var insideItem = false
        var articleId: String = null
        var articleTitle: String = null
        var articleByline: String = null
        var articleDateline: String = null
        var articleText: String = null

        val numArticles = new AtomicInteger(0)

        var skipDocument = false

        while (xml.hasNext) {
          val event = xml.nextEvent()

          if (event.isStartElement && !skipDocument) {
            val element = event.asStartElement()

            if (element.getName.getLocalPart == "DOC") {
              assert(!insideItem, s"starting document while inside $articleId")
              insideItem = true
              articleId = null
              articleTitle = "n/a"
              articleByline = "n/a"
              articleDateline = "n/a"
              articleText = null
              firstHead = true
              firstBy = true
              firstDate = true
              firstText = true
            } else if (element.getName.getLocalPart == "DOCNO") {
              assert(insideItem, s"encountered docno while outside an item")
              articleId = xml.getElementText
              skipDocument = corpus.canLoad(articleId)
              if (skipDocument) {
                log.debug(s"Skipping document $articleId")
              }
            } else if (element.getName.getLocalPart == "HEAD" || element.getName.getLocalPart == "HL" && firstHead) {
              articleTitle = xml.getElementText
              firstHead = false
            } else if (element.getName.getLocalPart == "BYLINE" && firstBy) {
              articleByline = xml.getElementText
              firstBy = false
            } else if (element.getName.getLocalPart == "DATELINE" && firstDate) {
              articleDateline = xml.getElementText
              firstDate = false
            } else if (element.getName.getLocalPart == "TEXT") {
              if (firstText) {
                articleText = xml.getElementText
              } else {
                articleText += "\n" + xml.getElementText
              }
              firstText = false
            }
          } else if (event.isEndElement) {
            val element = event.asEndElement()

            if (element.getName.getLocalPart == "DOC") {
              assert(insideItem, s"encountered end of document $articleId without being in a document")
              if (!skipDocument) {
                corpusService.submit{
                  for (document <- managed(Document.fromString[NewsArticle](articleText))) {
                    document(BaseDocument.id) = articleId
                    document(HasTitle.title) = articleTitle
                    document(NewsArticle.byline) = articleByline
                    document(NewsArticle.dateline) = articleDateline
                    try {
                      corpus.save(document)
                    } catch {
                      case e: NullPointerException =>
                        log.error("Failed to parse document {}", document.describe())
                        throw Throwables.propagate(e)
                    }
                  }
                  numArticles.incrementAndGet()
                  plog.info("saved document #{}: |{}|", articleId, articleTitle)
                }
              }
              insideItem = false
              skipDocument = false
            }
          }
        }
        plog.update("enumerated {} articles from {}", numArticles, file)
      }
    }
  }

  corpusService.shutdown()
  corpusService.awaitTermination(1l, TimeUnit.HOURS)
}
