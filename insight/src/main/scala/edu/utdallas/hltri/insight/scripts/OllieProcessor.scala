package edu.utdallas.hltri.insight.scripts

import java.io.BufferedReader
import java.nio.file.{Paths, Files}
import java.text.DecimalFormat

import edu.knowitall.common.Timing
import edu.knowitall.common.Resource.using
import edu.knowitall.ollie.{OllieExtractionInstance, Ollie}
import edu.knowitall.ollie.confidence.{OllieFeatureSet, OllieConfidenceFunction}
import edu.knowitall.openparse.OpenParse
import edu.knowitall.tool.segment.Segmenter

import scala.collection.JavaConverters._

sealed abstract class OutputFormat {
  def header: Option[String]
  def format(conf: Double, extr: OllieExtractionInstance): String

object OutputFormat {
  val confFormatter = new DecimalFormat("#.###")
  def parse(format: String): OutputFormat = {
    format.toLowerCase match {
      case "tabbed" => TabbedFormat
    }
  }
}
case object TabbedFormat extends OutputFormat {
  def headers = Seq("confidence", "arg1", "rel", "arg2", "enabler", "attribution", "text", "pattern", "dependencies")
  def header = Some(headers.mkString("\t"))
  def format(conf: Double, inst: OllieExtractionInstance): String =
    Iterable(OutputFormat.confFormatter.format(conf),
      inst.extr.arg1.text,
      inst.extr.rel.text,
      inst.extr.arg2.text,
      inst.extr.enabler.map(_.text),
      inst.extr.attribution.map(_.text),
      inst.sent.text,
      inst.pat,
      inst.sent.serialize).mkString("\t")
}

/**
 * Created by travis on 8/4/15.
 */
object OllieProcessor extends App {
  val inFile = Paths.get(args(0))
  val outFile = Paths.get(args(0) + ".conll")


  System.err.println("Loading parser models... ")
  val parser = Timing.timeThen {
    new StanfordNNParser()
  } { ns =>
    System.err.println(Timing.Seconds.format(ns))
  }

  System.err.print("Loading ollie models... ")
  val ollieExtractor = Timing.timeThen {
    val configuration =
      new OpenParse.Configuration(
        confidenceThreshold = 0.005)

    val openparse = OpenParse.fromModelUrl(OpenParse.defaultModelUrl, configuration)
    new Ollie(openparse)
  } { ns =>
    System.err.println(Timing.Seconds.format(ns))
  }

  System.err.print("Loading ollie confidence function... ")
  val confFunction = Timing.timeThen {
    OllieConfidenceFunction.fromUrl(OllieFeatureSet, OllieConfidenceFunction.defaultModelUrl)
  } { ns =>
    System.err.println(Timing.Seconds.format(ns))
  }

  System.err.print("Loading opennlp sentence splitter... ")
  val sentencer: Segmenter = Timing.timeThen {
    System.err.println("Prose input split by OpenNlpSentencer.");
    ??? //new OpenNlpSentencer()
  } { ns =>
    System.err.println(Timing.Seconds.format(ns))
  }


  using(Files.newBufferedWriter(outFile)) { writer =>
    writer.println(header)

    // process a source and output extractions
    def processSource(source: BufferedReader): Unit = {
      val ns = Timing.time {
        val lines: Iterator[String] = parseLines(source.lines().iterator().asScala, sentencer) filter (!_.isEmpty)
        // group the lines so we can parallelize
        for (group <- lines.grouped(10000);
             sentence: String <- group.par) {
            // parse the sentence
            val graph = parser.dependencyGraph(sentence)

              // extract sentence and compute confidence
              val extrs = ollieExtractor.extract(graph).iterator.map(extr => (confFunction.getConf(extr), extr))

              extrs match {
                case it if it.isEmpty =>
                case extrs => (extrs filter (_._1 >= 0.005)).toList.sortBy(-_._1).foreach {
                  case (conf, e) =>
                    writer.write(TabbedFormat.format(conf, e))
                    writer.newLine()
                    writer.flush()
                }
              }
            }
          }
      System.err.println()
      System.err.println("Completed in " + Timing.Seconds.format(ns) + " seconds")
    }

    System.err.println("\nRunning extractor on multiple files...")
    val ns = Timing.time {
      for ((file, i) <- files.iterator.zipWithIndex) {
        System.err.println("Processing file " + file + " (" + (i+1) + "/" + files.size + ")...")
        System.err.println()
        using(Source.fromFile(file, settings.encoding)) { source =>
          processSource(source)
        }
      }
    }
    System.err.println("All files completed in " + Timing.Seconds.format(ns) + " seconds")
  }


  def parseLines(lines: Iterator[String], sentencer: Segmenter) = {
    new SentenceIterator(sentencer, lines.buffered)
  }
}

class SentenceIterator(sentencer: Segmenter, private var lines: BufferedIterator[String]) extends Iterator[String] {
  var sentences: Iterator[String] = Iterator.empty

  lines.dropWhile(_.trim.isEmpty)

  def nextSentences = {
    val (paragraph, rest) = lines.span(!_.trim.isEmpty)
    lines = rest.dropWhile(_.trim.isEmpty).buffered
    sentencer.segmentTexts(paragraph.mkString(" ")).iterator.buffered
  }

  def hasNext: Boolean = {
    if (sentences.hasNext) {
      true
    }
    else if (!lines.hasNext) {
      false
    }
    else {
      sentences = nextSentences
      sentences.hasNext
    }
  }

  def next: String = {
    if (sentences.hasNext) {
      sentences.next()
    }
    else {
      sentences = nextSentences
      sentences.next()
    }
  }
}
