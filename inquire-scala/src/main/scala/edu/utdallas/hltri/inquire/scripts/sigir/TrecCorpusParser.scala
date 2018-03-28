package edu.utdallas.hltri.inquire.scripts.sigir

import java.io._
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPInputStream
import javax.xml.stream.{XMLEventReader, XMLInputFactory}

import edu.utdallas.hltri.concurrent.BoundedThreadPool
import edu.utdallas.hltri.framework.ProgressLogger
import edu.utdallas.hltri.inquire.documents.TRECDocument
import edu.utdallas.hltri.io.IOUtils
import edu.utdallas.hltri.scribe.io.JsonCorpus
import edu.utdallas.hltri.scribe.text.{BaseDocument, Document}

import scala.collection.JavaConverters._

/**
 * Created by travis on 11/6/15.
 */
object TrecCorpusParser extends App with Logging {
  val srcs = args.slice(0, args.length - 1)
  val dst = args.last

  val corpus = JsonCorpus.at[TRECDocument](dst).tiered().build()

  val corpusService = BoundedThreadPool.create(32)
  implicit def runnable(f: => Any): Runnable = new Runnable() { def run() = f }

  val xmlFactory = XMLInputFactory.newInstance()

  def getNestedText(xml: XMLEventReader, tag: String): String = {
    val value = new StringBuilder()
    var next = xml.nextEvent()
    while (!next.isEndElement || next.asEndElement().getName.getLocalPart != tag) {
      if (next.isCharacters) {
        if (value.nonEmpty) {
          value.append("\n")
        }
        value.append(next.asCharacters().getData)
      }
      next = xml.nextEvent()
    }
    value.toString.trim
  }

  def readElementBody(xml: XMLEventReader): String = {
    val buffer = new StringWriter(1024)
    var depth = 1
    while (xml.hasNext) {
      val event = xml.peek()

      event.writeAsEncodedUnicode(buffer)

      if (event.isStartElement) {
        depth += 1
      } else if (event.isEndElement) {
        depth -= 1
        if (depth == 0) {
          return buffer.getBuffer.toString
        }
      }

      xml.nextEvent()
    }
    throw new IOException("Failed to parse XML")
  }

  for (plog <- managed(ProgressLogger.indeterminateSize("parsing XML", 1, TimeUnit.MINUTES))) {
    for (src <- srcs; file <- IOUtils.`lazy`.iterate(src).asScala if file.getName.endsWith(".xml.gz")) {
      for (istream <- managed(new InputStreamReader(new GZIPInputStream(new FileInputStream(file)))); xml <- managed(
        xmlFactory.createXMLEventReader(istream))) {

        while (xml.hasNext()) {
          val event = xml.nextEvent()
          if (event.isStartElement()) {
            val start = event.asStartElement()
            val name = start.getName().getLocalPart()

            if ("DOC".equals(name)) {
              val content = "<DOC>" + readElementBody(xml);
              corpusService.submit(new Runnable {
                override def run(): Unit = {
//                  log.info(s"Parsing: $content")
                  val doc = Document.fromStringWithMime[TRECDocument](content, "text/xml")
                  val id = gate.Utils.cleanStringFor(doc.asGate, doc.asGate.getAnnotations("Original markups").get("DOCNO"))
                  doc.set(BaseDocument.id, id)
                  corpus.save(doc)
                  doc.close()
                  plog.update(s"Parsed document $id")
                }
              })
            }
          }
        }
      }
    }
  }

  corpusService.shutdown()
  corpusService.awaitTermination(1l, TimeUnit.HOURS)
}
