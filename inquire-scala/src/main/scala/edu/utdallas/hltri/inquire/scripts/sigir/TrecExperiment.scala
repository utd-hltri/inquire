package edu.utdallas.hltri.inquire.scripts.sigir

import java.io.{BufferedReader, File, FileInputStream, InputStreamReader}
import java.nio.file.{Files, Paths}
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPInputStream

import edu.utdallas.hltri.framework.ProgressLogger
import edu.utdallas.hltri.inquire.documents.TRECDocument
import edu.utdallas.hltri.inquire.eval.TrecRunWriter
import edu.utdallas.hltri.inquire.lucene.similarity.Similarities
import edu.utdallas.hltri.inquire.lucene.{EagerDocumentFactory, LuceneSearchEngine}
import edu.utdallas.hltri.io.IOUtils
import edu.utdallas.hltri.scribe.text.{BaseDocument, Document}
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.benchmark.quality.trec.TrecTopicsReader
import org.apache.lucene.document.{Document => lDocument}
import org.apache.lucene.queryparser.classic.{QueryParser, QueryParserBase}
import org.apache.lucene.search.{BooleanClause, BooleanQuery, Query}

import scala.collection.JavaConverters._

/**
 * Created by travis on 12/10/15.
 */

object TrecExperimentUtils {
  def loadIndex(path: String, indexField: String = "text"): LuceneSearchEngine[Document[TRECDocument]] = {
    new LuceneSearchEngine[Document[TRECDocument]](path,
      new EnglishAnalyzer(/**/),
      indexField,
      new EagerDocumentFactory[Document[TRECDocument]] {
        override def build(t: lDocument, u: Int): Document[TRECDocument] = {
          val d: Document[TRECDocument] = Document.fromString[TRECDocument](t.get(indexField))
          d(BaseDocument.id) = t.get("id")
          d
        }
      })
  }

  def loadTopics(path: String, queryFields: Set[String] = Set("title"), indexField: String = "text"): Seq[(Query, String)] = {
    val ttReader = new TrecTopicsReader

    val bReader = if (path.endsWith(".gz")) new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(new File(path)))))
    else Files.newBufferedReader(Paths.get(path))

    val qqs = managed(bReader).acquireAndGet(reader => ttReader.readQueries(reader))


    val qp: QueryParser = new QueryParser(indexField, new EnglishAnalyzer())

    for (qq <- qqs) yield {
      val bq = new BooleanQuery
      for (field <- queryFields) {
        bq.add(qp.parse(QueryParserBase.escape(qq.getValue(field))), BooleanClause.Occur.SHOULD)
      }
      bq -> qq.getQueryID
    }
  }

  def parseQuerySpec(fieldSpec: String): Set[String] = {
    var fieldSet = Set[String]()
    if (fieldSpec.indexOf('T') >= 0) fieldSet += "title"
    if (fieldSpec.indexOf('D') >= 0) fieldSet += "description"
    if (fieldSpec.indexOf('N') >= 0) fieldSet += "narrative"
    fieldSet
  }
}

object TrecExperiment extends App with Logging {
  if (args.length < 4 || args.length > 5) {
    System.err.println("Usage: TrecExperiment <indexFile> <topicFile> <similarity> <submissionFile> [querySpec]")
    System.err.println("indexFile: index directory")
    System.err.println("topicFile: input file containing TREC topics")
    System.err.println("similarity: similarity function to use (i.e. BM25, TFIDF, DFR, IB, LMD, LMJM)")
    System.err.println("submissionFile: output file for submission")
    System.err.println("querySpec: string composed of fields to use in query consisting of T=title,D=description,N=narrative:")
    System.err.println("\texample: TD (query on Title + Description). The default is T (title only)")
    System.exit(1)
  }

  val index = TrecExperimentUtils.loadIndex(args(0))
  val topics = TrecExperimentUtils.loadTopics(args(1))
  val similarity = Similarities.valueOf(args(2).toUpperCase).similarity
  val submissionFile = Paths.get(args(3))
  val runtag = IOUtils.removeExtension(submissionFile.getFileName.toString)
  val querySpec = TrecExperimentUtils.parseQuerySpec(args.applyOrElse(4, (i: Int) => "T"))

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
