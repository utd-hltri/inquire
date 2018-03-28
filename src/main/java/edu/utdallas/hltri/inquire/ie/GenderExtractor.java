package edu.utdallas.hltri.inquire.ie;

import java.io.Serializable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.utdallas.hltri.func.CloseableFunction;
import edu.utdallas.hltri.logging.Logger;

/**
 *
 * @author travis
 */
public class GenderExtractor implements CloseableFunction<CharSequence, GenderExtractor.Gender> {
  public static enum Gender implements Serializable {
    MALE("man", "men",
      "male", "males",
      "boy", "boys",
      "dude", "dudes",
      "gentleman", "gentlemen",
      "guy", "guys",
      "lad", "lads",
      "he", "him", "his"),

    FEMALE("woman", "women",
      "female", "females",
      "girl", "girls",
      "dudette", "dudettes",
      "lady", "ladies",
      "gal", "gals",
      "lass", "lasses", "lassie", "lassies",
      "she", "her", "hers");

    private final Pattern pattern;

    public final String[] lexicon;

    private Gender(String... elements) {
      this.lexicon = elements;
      StringBuilder string = new StringBuilder();

      string.append("(?i)(");
      for (int i = 0; i < elements.length; i++) {
        if (i > 0) { string.append("|"); }
        string.append("\\b").append(Pattern.quote(elements[i])).append("\\b");
      }
      string.append(")");
      this.pattern = Pattern.compile(string.toString());
    }

    public Gender getOpposite() {
      switch(this) {
        case MALE:
          return FEMALE;
        case FEMALE:
          return MALE;
        default:
          return null;
      }
    }


    public Pattern toPattern() {
      return this.pattern;
    }

    @Override
    public String toString() {
      switch(this) {
        case MALE:
          return "MALE";
        case FEMALE:
          return "FEMALE";
        default:
          return "";
      }
    }
  }

  final Logger log = Logger.get(GenderExtractor.class);

  public Gender extract(CharSequence topic) {
    String phrase = topic.toString().toLowerCase();
    Gender gender = null;

    for (Gender GENDER : Gender.values()) {
      Matcher matcher = GENDER.toPattern().matcher(phrase);
      if (matcher.find()) {
        log.trace("Found {} from match {} in {}.", GENDER, matcher.group(0), phrase);
        if (gender != null) {
          gender = null;
          break;
        }
        else
          gender = GENDER;
      }
    }

    if (gender != null) {
      log.info("Found gender requirement: {}.", gender);
    }

    return gender;
  }

  @Override
  public Gender apply(CharSequence charSequence) {
    return extract(charSequence);
  }
}
