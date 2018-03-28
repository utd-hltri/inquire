package edu.utdallas.hltri.inquire.scripts.sigir

import java.nio.file.{Files, Paths}
import java.util.concurrent.TimeUnit

import edu.utdallas.hltri.framework.ProgressLogger
import edu.utdallas.hltri.inquire.eval.TrecRunWriter
import edu.utdallas.hltri.inquire.lucene.similarity.Similarities
import edu.utdallas.hltri.io.IOUtils
import edu.utdallas.hltri.scribe.text.BaseDocument
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.document.{Document => lDocument}
import org.apache.lucene.queryparser.classic.{QueryParser, QueryParserBase}
import org.apache.lucene.search.Query

import scala.collection.JavaConverters._


/**
 * Created by travis on 12/10/15.
 */

object SilverExperimentUtils {
  def loadTopics(path: String, indexField: String = "text"): Seq[(Query, String)] = {
    (for (line <- Files.readAllLines(Paths.get(path)).asScala) yield {
      val delim = line.indexOf('\t')
      val qid = line.substring(0, delim)
      val qp: QueryParser = new QueryParser(indexField, new EnglishAnalyzer())
      val query = qp.parse(QueryParserBase.escape(line.substring(delim + 1)))
      query -> qid
    }).toSeq
  }
}

object SilverExperiment extends App with Logging {
  if (args.length < 4 || args.length > 5) {
    System.err.println("Usage: SilverExperiment <indexFile> <topicFile> <similarity> <submissionFile> [maxHits]")
    System.err.println("indexFile: index directory")
    System.err.println("topicFile: input file containing TREC topics")
    System.err.println("similarity: similarity function to use (i.e. BM25, TFIDF, DFR, IB, LMD, LMJM)")
    System.err.println("submissionFile: output file for submission")
    System.err.println("maxHits: optional bound for number of search results per query (default = 1000)")
    System.exit(1)
  }

  val index = TrecExperimentUtils.loadIndex(args(0))
  val topics =  SilverExperimentUtils.loadTopics(args(1))
  val similarity = Similarities.valueOf(args(2).toUpperCase).similarity
  val submissionFile = Paths.get(args(3))
  val runtag = IOUtils.removeExtension(submissionFile.getFileName.toString)
  val maxHits = args.applyOrElse(4, (i: Int) => "1000").toInt

  for (qrels <- managed(new TrecRunWriter(submissionFile));
       plog <- managed(ProgressLogger.fixedSize("retrieving", topics.size, 1l, TimeUnit.SECONDS))) {
    for ((topic, qid) <- topics) {
      log.info(s"Retrieving topic ${topic}")
      for (result <- index.search(similarity, topic, 1000).getResults.asScala) {
        qrels
          .writeQRel(qid,
            result.getValue.get(BaseDocument.id),
            result.getRank,
            result.getScore,
            runtag)
      }
      plog.update(s"retrieved $qid")
    }
  }
}
