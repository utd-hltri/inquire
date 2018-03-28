

name := "inquire-scala"

version := "0.1"

organization := "edu.utdallas.hltri"

// enable publishing to maven
publishMavenStyle := true

scalaVersion := "2.11.2"

// Connects STDIN to sbt during forked runs
connectInput in run := true

// Get rid of output prefix
outputStrategy in run := Some(StdoutOutput)

// When  using sbt-run, fork to a new process instead of running within the sbt process
fork in run := true

// Set default java options: enable assertions, set memory, set server mode
javaOptions ++= Seq("-ea", "-esa", "-Xmx14g", "-server")

// Set javac options
javacOptions ++= Seq("-source", "1.8", "-target", "1.8", "-Xlint:unchecked")

// Always export a .jar rather than .class files
exportJars := true

libraryDependencies += "org.scala-lang.modules" %% "scala-java8-compat" % "0.7.0"

// Fancy scala resource management
libraryDependencies += "com.jsuereth" %% "scala-arm" % "1.4"

lazy val `inquire` = RootProject(file(".."))

lazy val `inquire-scala` = Project(
  id = "inquire-scala",
  base = file("."))
  .dependsOn(util)