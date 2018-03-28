package edu.utdallas.hltri.inquire.scripts.sigir

import java.nio.file.{Files, Paths}

import edu.utdallas.hltri.io.IOUtils
import org.apache.commons.math3.linear.RealVectorFormat

import scala.collection.JavaConverters._


/**
 * Created by travis on 12/4/15.
 */
object Convert2Norms extends App {
  val inputFile = args(0)
  val outputFile = args.applyOrElse(1, (i: Int) => inputFile + ".unit")
  val weights = scala.collection.mutable.Buffer[Double]()
  val format = new RealVectorFormat("", " ", "")
  for (reader <- managed(Files.newBufferedReader(Paths.get(inputFile)));
       writer <- managed(Files.newBufferedWriter(Paths.get(outputFile)))) {
    for (inLine <- IOUtils.`lazy`.readLines(reader).asScala) {
      val delim = inLine.indexOf(' ')
      val word = inLine.substring(0, delim)
      val vector = format.parse(inLine.substring(delim + 1))
      vector.unitize()
      writer.write(s"$word ${format.format(vector)}\n")
    }
  }
}
