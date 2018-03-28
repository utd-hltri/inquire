name := "insight-ollie"

version := "1.0"

scalaVersion := "2.10.4"

// Ollie open information extraction system
libraryDependencies += "edu.washington.cs.knowitall.ollie" %% "ollie-core" % "1.0.+"

// University of Washington NLP tools
libraryDependencies ++= Seq(
  "edu.washington.cs.knowitall.nlptools" %% "nlptools-core" % "2.4.+",
  "edu.washington.cs.knowitall.nlptools" %% "nlptools-parse-stanford" % "2.4.+"
)

// Update stanford CoreNLP junk
libraryDependencies ++= Seq(
  "edu.stanford.nlp" % "stanford-corenlp" % "3.5.2",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.5.2" classifier "models"
)
