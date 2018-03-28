package edu.utdallas.hltri.inquire.scripts.sigir

import java.nio.file.{Files, Paths}
import java.util.concurrent.{ThreadLocalRandom, TimeUnit}

import edu.utdallas.hltri.framework.ProgressLogger
import edu.utdallas.hltri.inquire.ANN
import edu.utdallas.hltri.inquire.ANN.LinearIndex
import edu.utdallas.hltri.inquire.documents.TRECDocument
import edu.utdallas.hltri.inquire.lucene.{EagerDocumentFactory, LuceneSearchEngine}
import edu.utdallas.hltri.scribe.annotators.StanfordCoreAnnotator
import edu.utdallas.hltri.scribe.io.JsonCorpus
import edu.utdallas.hltri.scribe.text.Document
import edu.utdallas.hltri.scribe.text.annotation.Token
import org.apache.commons.math3.distribution.EnumeratedDistribution
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.document.{Document => lDocument}

import scala.collection.JavaConverters._

object MVNTest extends App {

  val corpus1File = "/home/travis/work/collections/trec/adhoc_vol4"
  val corpus2File = "/home/travis/work/collections/trec/adhoc_vol5"

  val corpus1 = JsonCorpus.at[TRECDocument](corpus1File).tiered().build()
  val corpus2 = JsonCorpus.at[TRECDocument](corpus2File).tiered().build()

  lazy val ann = new ANN("/shared/aifiles/disk1/travis/data/models/GloVe/glove.6B.300d.txt", new LinearIndex)

  val index = new LuceneSearchEngine[Document[TRECDocument]](
    "/shared/aifiles/disk1/travis/data/indices/trec_adhoc_vols_4_5.idx",
    new EnglishAnalyzer(/**/),
    "text",
    new EagerDocumentFactory[Document[TRECDocument]]{
      override def build(t: lDocument, u: Int): Document[TRECDocument] = corpus.load(t.get("id"))
    })

  def printWords(vector: Array[Double]) = {
    System.out.println(ann.getNearest(vector, 3).asScala.map(_.toString).mkString("\n"))
  }

//  val corpusMean: Array[Double] = IOUtils.readObject(new File(corpusFile + ".means.ser"));
//  val corpusCovariance: Array[Array[Double]] = IOUtils.readObject(new File(corpusFile + ".covariance.ser"));
//  val corpusMVN = new MultivariateNormalDistribution(corpusMean, corpusCovariance)



  lazy val corpus1Ids = corpus1.getIdStream.iterator().asScala.toArray
  lazy val corpus2Ids = corpus2.getIdStream.iterator().asScala.toArray

  lazy val corpus1Size = 201782 //corpus1Ids.size
  lazy val corpus2Size = 257743 //corpus2Ids.size

  def randomDocument(): String = {
    val rid = ThreadLocalRandom.current().nextLong(corpus1Size + corpus2Size)

    if (rid >= corpus1Size) {
      corpus2Ids((rid - corpus1Size).toInt)
    } else {
      corpus1Ids(rid.toInt)
    }
  }

  val query = new StringBuilder()

  val output = Paths.get(args(0))
  val nQueries = args(1).toInt

  lazy val stanfordAnnotator = new StanfordCoreAnnotator.Builder[TRECDocument]().tokenize().build();

  def loadOrGetTokens(doc: Document[TRECDocument]): Iterable[Token] = {
    var tokens = doc.get("stanford", Token.TYPE)
    if (tokens.isEmpty) {
      stanfordAnnotator.annotate(doc)
      doc.sync()
      tokens = doc.get("stanford", Token.TYPE)
    }
    tokens.asScala
  }

  for (queryWriter <- managed(Files.newBufferedWriter(output.resolveSibling(output.getFileName + ".queries.tsv")));
       qrelsWriter <- managed(Files.newBufferedWriter(output.resolveSibling(output.getFileName + ".qrels.tsv")));
       plog <- managed(ProgressLogger.fixedSize("generating queries", 1000, 10, TimeUnit.SECONDS))) {
    var q = 0
    while (q < nQueries) {
      val relDocId = randomDocument()
      val irrDocId = randomDocument()

      for (doc <- managed(if (corpus1.canLoad(relDocId)) corpus1.load(relDocId) else corpus2.load(relDocId))) {
        val words = for (token <- loadOrGetTokens(doc);
                         word = token.toCollapsedLowercase if ann.vectors.asScala.contains(word)) yield word.toLowerCase
        val tfs: Map[String, Int] = words.groupBy(i => i).mapValues(_.size)
        System.out.println(tfs)
        val idfs: java.util.List[org.apache.commons.math3.util.Pair[String, java.lang.Double]] = tfs.map{case (word: String, tf: Int) =>
          org.apache.commons.math3.util.Pair.create(word, Double.box(tf * index.getNormalizedInverseDocumentFrequency(index.newPhraseQuery("text", word))))
        }.filterNot(_.getSecond.isInfinite).toList.asJava
        for (idf <- idfs.asScala) {
          System.out.println(idf);
        }
//
//
//
//
//          val vector = ann.vectors(word)
//          mean.increment(vector)
//          covariance.increment(vector)
//        }
//
//
//        val means = doc.get(MVN.wordMeans)
//        val covariances = doc.get(MVN.wordCovariances)
//        for (i <- means.indices) {
////          means(i) -= corpusMean(i)
//          for (j <- means.indices) {
////            covariances(i)(j) -= corpusCovariance(i)(j)
//          }
//        }
        query.setLength(0)

//        query.append(ann.getNearest(means, 3))



        try {
//          val mvn = new MultivariateNormalDistribution(means, covariances)
          val dist = new EnumeratedDistribution[String](idfs)

          val nKeywords = 3 // sample from Zipf
          for (i <- 0 until nKeywords) {
            if (i > 0) {
              query.append(' ')
            }

            val nWords = 1 // sample from Zipf
            for (j <- 0 until nWords) {
              if (j > 0) {
                query.append(' ')
              }

//              while (mvn.density(word) > 0.5) {
//                System.out.println("Ignoring word " + ann.getNearest(word, 1))
//                word = mvn.sample()
//              }
//              for (l <- word.indices) {
//                word(l) += corpusMean(l)
//              }

              val word: String = if (math.random < 0.10) {
                ann.getNearest(dist.sample(), 2).asScala.last.getValue
              } else {
                dist.sample()
              }

              query.append(word)
            }
          }
          queryWriter.write(s"$q\t$query\n")
          qrelsWriter.write(s"$q\t$relDocId\t1\n")
          qrelsWriter.write(s"$q\t$irrDocId\t1\n")

          if (q % 100 == 0) {
            queryWriter.flush()
            qrelsWriter.flush()
          }

          q += 1

          plog.update(s"generated query $q: |$query|")
        } catch {
          case ex: org.apache.commons.math3.linear.SingularMatrixException =>
//            val nwords = doc.get("stanford", Token.TYPE).map(_.toString).toSet.size
//            val nfeats = means.length
//            if (nwords >= nfeats) {
//              System.err.println("Failed to construct MVN")
//              System.err.println("Means = " + java.util.Arrays.toString(means))
//              System.err.println("Covariances = " + covariances.map(java.util.Arrays.toString _).mkString)
//              System.err.println("Document = " + doc.describe())
//              System.err.println("Number of words = " + doc.get("stanford", Token.TYPE).size)
//            }
//            System.err.println(s"Skipping $relDocId with $nwords < $nfeats words")
        }
      }
    }
  }
}