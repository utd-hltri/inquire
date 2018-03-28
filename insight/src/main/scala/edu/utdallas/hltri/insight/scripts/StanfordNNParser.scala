package edu.utdallas.hltri.insight.scripts

import edu.knowitall.tool.parse.BaseStanfordParser.CollapseType
import edu.knowitall.tool.parse.graph.{DependencyGraph, Dependency}
import edu.knowitall.tool.parse.{StanfordParser, ConstituencyParser, BaseStanfordParser}
import edu.knowitall.tool.postag.{PostaggedToken, StanfordPostagger, Postagger}
import edu.stanford.nlp.ling.TaggedWord
import edu.stanford.nlp.parser.lexparser.LexicalizedParser
import edu.stanford.nlp.parser.nndep.DependencyParser

/**
 * Created by travis on 8/4/15.
 */
class StanfordNNParser(lp: LexicalizedParser, val postagger: Postagger) extends BaseStanfordParser with ConstituencyParser {
  def this(postagger: Postagger = new StanfordPostagger()) = this(DependencyParser.loadFromModelFile(DependencyParser.DEFAULT_MODEL), postagger)

  override def dependencies(string: String, collapse: CollapseType): Iterable[Dependency] = {
    val tokens = postagger.postag(string)
    StanfordParser.dependencyHelper(lp.parse(postagToStanfordRepr(tokens)), collapse)._2
  }

  private def postagToStanfordRepr(tokens: Seq[PostaggedToken]): java.util.List[TaggedWord] = {
    val words = new java.util.ArrayList[TaggedWord](tokens.size)

    tokens.foreach { token =>
      val w = new TaggedWord(token.string, token.postag)
      w.setBeginPosition(token.offsets.start)
      w.setEndPosition(token.offsets.end)
      words.add(w)
    }

    words
  }

  override def dependencyGraphPostagged(tokens: Seq[PostaggedToken], collapse: CollapseType) = {
    val (nodes, deps) = StanfordParser.dependencyHelper(lp.parse(postagToStanfordRepr(tokens)), collapse)
    new DependencyGraph(nodes.toList.sortBy(_.indices), deps)
  }

  override def parse(string: String) = {
    val tokens = postagger.postag(string)
    StanfordParser.convertTree(lp.parse(postagToStanfordRepr(tokens)))
  }
}