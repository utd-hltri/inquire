package edu.utdallas.hltri.inquire.lucene.similarity;

import org.apache.lucene.search.similarities.AfterEffectB;
import org.apache.lucene.search.similarities.AxiomaticF2EXP;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BasicModelIne;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.DFISimilarity;
import org.apache.lucene.search.similarities.DFRSimilarity;
import org.apache.lucene.search.similarities.DistributionSPL;
import org.apache.lucene.search.similarities.IBSimilarity;
import org.apache.lucene.search.similarities.IndependenceChiSquared;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.LambdaDF;
import org.apache.lucene.search.similarities.NormalizationZ;
import org.apache.lucene.search.similarities.Similarity;

/**
 * Created by travis on 5/29/15.
 */
public enum Similarities {
  BM25(new BM25Similarity()),
  TFIDF(new ClassicSimilarity()),
  DFR(new DFRSimilarity(new BasicModelIne(), new AfterEffectB(), new NormalizationZ())),
  IB(new IBSimilarity(new DistributionSPL(), new LambdaDF(), new NormalizationZ())),
  LMD(new LMDirichletSimilarity()),
  LMJM(new LMJelinekMercerSimilarity(0.5f)),
  F2EXP(new AxiomaticF2EXP()),
  DFI(new DFISimilarity(new IndependenceChiSquared()));

  public final Similarity similarity;

  Similarities(final Similarity similarity) {
    this.similarity = similarity;
  }
}
