name := "inquire-med"

version := "0.1.0"

organization := "edu.utdallas.hltri"

libraryDependencies += "net.sf.trove4j" % "trove4j" % "3.0.3"

libraryDependencies ++= Seq(
  "edu.utdallas.hltri" % "medbase" % "1.0.0",
  "edu.utdallas.hltri" % "insight-wiki" % "1.0.1"
)

// enable publishing to maven
publishMavenStyle := true

// do not append scala version to the generated artifacts
crossPaths := false

// do not add scala libraries as a dependency!
autoScalaLibrary := false
