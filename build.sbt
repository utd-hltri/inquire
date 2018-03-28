import sbt.ProjectRef

name := "inquire"

version := "0.1.0"

organization := "edu.utdallas.hltri"

description := "Information Retrieval (IR) and Question Answering (Q&A) Project"

publishTo := sonatypePublishTo.value

// enable publishing to maven
publishMavenStyle := true

// do not append scala version to the generated artifacts
crossPaths := false

// do not add scala libraries as a dependency!
autoScalaLibrary := false

lazy val commonSettings = Seq(
  // Connects STDIN to sbt during forked runs
  connectInput in run := true,
  // Get rid of output prefix
  outputStrategy in run := Some(StdoutOutput),
  // When  using sbt-run, fork to a new process instead of running within the sbt process
  fork in run := true,
  // Set default java options: enable assertions, set memory, set server mode
  javaOptions ++= Seq("-ea", "-esa", "-Xmx14g", "-server"),
  // Set javac options
  javacOptions ++= Seq("-source", "1.8", "-target", "1.8", "-Xlint:unchecked"),
  // Always export a .jar rather than .class files
  exportJars := true
)

lazy val luceneVersion = "6.6.+"

// Lucene IR library
libraryDependencies ++= Seq(
  "org.apache.lucene" % "lucene-core" % luceneVersion,
  "org.apache.lucene" % "lucene-queryparser" % luceneVersion,
  "org.apache.lucene" % "lucene-analyzers-common" % luceneVersion,
  "org.apache.lucene" % "lucene-backward-codecs" % luceneVersion,
  "org.apache.lucene" % "lucene-queries" % luceneVersion,
  "org.apache.lucene" % "lucene-benchmark" % luceneVersion
)

// Adds @Notnull annotation
libraryDependencies += "com.google.code.findbugs" % "jsr305" % "3.0.+"

// XML parser
libraryDependencies += "org.jdom" % "jdom2" % "2.0.5"

libraryDependencies += "org.apache.commons" % "commons-math3" % "3.5"

libraryDependencies += "org.apache.commons" % "commons-lang3" % "3.4"

libraryDependencies += "net.sf.trove4j" % "trove4j" % "3.0.3"

libraryDependencies += "cc.mallet" % "mallet" % "2.0.7"

// Simple CLI library
libraryDependencies += "info.picocli" % "picocli" % "0.9.7"

// hltri libraries
libraryDependencies ++= Seq(
  "edu.utdallas.hltri" % "hltri-util" % "1.0.1",
  "edu.utdallas.hltri" % "hltri-ml-api" % "1.0.0",
  "edu.utdallas.hltri" % "scribe" % "0.3.1"
)

lazy val inquire = (project in file("."))
  .settings(commonSettings)

lazy val `inquire-med` = (project in file("inquire-med"))
  .dependsOn(inquire)
  .settings(commonSettings)
