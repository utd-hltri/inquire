package edu.utdallas.hltri.inquire.scripts.sigir

import java.io.File

import com.google.common.base.{CharMatcher, Splitter}
import edu.utdallas.hltri.inquire.documents.TRECDocument
import edu.utdallas.hltri.io.IOUtils
import edu.utdallas.hltri.scribe.io.{Corpus, JsonCorpus}
import edu.utdallas.hltri.scribe.text.annotation.Token
import edu.utdallas.hltri.scribe.text.{Document, DocumentAttribute}
import org.apache.commons.math3.stat.descriptive.moment.{VectorialCovariance, VectorialMean}

import scala.collection.JavaConverters._
import scala.collection.mutable


/**
  * Created by trg19 on 11/13/2015.
  */
class MVNComputer(vectorFile: String) {
  lazy val vectors: Map[String, Array[Double]] = {
    val splitter = Splitter.on(CharMatcher.WHITESPACE)
    val entries = for (line <- IOUtils.eager.readLines(vectorFile).asScala) yield {
      val it = splitter.split(line).iterator()
      val word = it.next()

      val values = mutable.Buffer[Double]()
      while (it.hasNext) {
        values += it.next().toDouble
      }

      word -> values.toArray
    }

    entries.toMap
  }

  lazy private[this] val D = vectors.valuesIterator.next().length

  def processCorpus(corpus: Corpus[TRECDocument], clean: Boolean = false): Unit = {
    for (id <- corpus.getIdStream.iterator().asScala) {
      for (doc <- managed(corpus.load(id))) {
        processDoc(doc, clean)
      }
    }
  }

  def processDoc(doc: Document[TRECDocument], clean: Boolean = false): Unit = {
    if (!doc.has(MVN.wordMeans) || !doc.has(MVN.wordCovariances) || clean) {
      val mean = new VectorialMean(D)
      val covariance = new VectorialCovariance(D, true)

      for (token <- doc.get("stanford", Token.TYPE).asScala;
           word = token.toCollapsedLowercase if vectors.contains(word)) {
        val vector = vectors(word)
        mean.increment(vector)
        covariance.increment(vector)
      }

      doc(MVN.wordMeans) = mean.getResult
      doc(MVN.wordCovariances) = covariance.getResult.getData
      doc.sync()
    }
  }
}

object MVN {
  val wordMeans = DocumentAttribute.inferred[TRECDocument, Array[Double]]("word_means")
  val wordCovariances = DocumentAttribute.inferred[TRECDocument, Array[Array[Double]]]("word_covariances")
}

object MVNComputer extends App {
  val corpus = JsonCorpus.at[TRECDocument](args(0)).tiered().build()
  val mvner = new MVNComputer(args(1))
  val clean = args.applyOrElse(2, (i: Int) => "false").toBoolean

  mvner.processCorpus(corpus, clean)
}

object MVNCollectionComputer extends App with Logging {
  val corpus = JsonCorpus.at[TRECDocument](args(0)).tiered().build()

  lazy val vectors: Map[String, Array[Double]] = {
    val splitter = Splitter.on(CharMatcher.WHITESPACE)
    val entries = for (line <- IOUtils.eager.readLines(args(1)).asScala) yield {
      val it = splitter.split(line).iterator()
      val word = it.next()

      val values = mutable.Buffer[Double]()
      while (it.hasNext) {
        values += it.next().toDouble
      }

      word -> values.toArray
    }

    entries.toMap
  }

  val D = vectors.valuesIterator.next().length

  val mean = new VectorialMean(D)
  val covariance = new VectorialCovariance(D, true)
  for (id <- corpus.getIdStream.iterator().asScala) {
    for (doc <- managed(corpus.load(id))) {
      for (token <- doc.get("stanford", Token.TYPE).asScala;
           word = token.toCollapsedLowercase if vectors.contains(word)) {
        val vector = vectors(word)
        mean.increment(vector)
        covariance.increment(vector)
      }
    }
  }

  log.info("Calculated means: " + java.util.Arrays.toString(mean.getResult))
  log.info("Calculated v-cv: " + covariance.getResult)


  IOUtils.saveObject(new File(args(0) + ".means.ser"): File, mean.getResult);
  IOUtils.saveObject(new File(args(0) + ".covariance.ser"), covariance.getResult.getData);
}