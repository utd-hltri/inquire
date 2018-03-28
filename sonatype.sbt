// Your profile name of the sonatype account. The default is the same with the organization value
sonatypeProfileName := "edu.utdallas.hltri"

// To sync with Maven central, you need to supply the following information:
publishMavenStyle := true

// License of your choice
licenses := Seq("APL2" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt"))

// Where is the source code hosted
import xerial.sbt.Sonatype._
sonatypeProjectHosting := Some(GitHubHosting("r-mal", "inquire", "ramon@hlt.utdallas.edu"))

