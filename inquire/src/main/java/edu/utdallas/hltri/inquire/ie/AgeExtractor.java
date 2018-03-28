package edu.utdallas.hltri.inquire.ie;

import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.utdallas.hltri.func.CloseableFunction;
import edu.utdallas.hltri.logging.Logger;

/**
 * @author travis
 */
public class AgeExtractor implements CloseableFunction<CharSequence, AgeExtractor.AgeRange> {

  private static final Logger                log              = Logger.get(AgeExtractor.class);
  /* Age Ranges
  935 **AGE[birth-12]
  1677 **AGE[in teens]
  1907 **AGE[90+]
  4762 **AGE[in 30s]
  4843 **AGE[in 20s]
  7440 **AGE[in 40s]
  9275 **AGE[in 50s]
  9579 **AGE[in 80s]
  9672 **AGE[in 60s]
  9863 **AGE[in 70s]
   */
  private final static int                   MIN_AGE          = 0;
  private final static int                   MAX_AGE          = 150;
  private final        Map<String, AgeRange> KNOWN_AGE_RANGES = new LinkedHashMap<String, AgeRange>() {{
    put("children", new AgeRange(0, 12));
    put("child", get("children"));
    put("adults", new AgeRange(20, MAX_AGE));
    put("adult", get("adults"));
    put("teenagers", new AgeRange(13, 19));
    put("teenager", get("teenagers"));
    put("seniors", new AgeRange(60, MAX_AGE));
    put("senior", get("seniors"));
    put("babies", new AgeRange(0, 2));
    put("baby", get("babies"));
    put("toddlers", new AgeRange(2, 4));
    put("toddler", get("toddlers"));
    put("infants", new AgeRange(0, 6));
    put("infant", get("infants"));
    put("teens", new AgeRange(13, 19));
    put("teen", get("teens"));
    put("newborns", new AgeRange(0, 1));
    put("newborn", get("newborns"));
    put("youths", new AgeRange(5, 12));
    put("youth", get("youths"));
    put("adolescents", new AgeRange(13, 19));
    put("adolescent", get("adolescents"));
    put("preschoolers", new AgeRange(4, 5));
    put("preschooler", get("preschoolers"));
    put("kids", new AgeRange(5, 16));
    put("kid", get("kids"));
    put("college aged", new AgeRange(19, 29));
    put("college age", get("college aged"));
    put("middle aged", new AgeRange(40, 59));
    put("middle age", get("middle aged"));
    put("young adults", new AgeRange(14, 19));
    put("young adult", get("young adults"));
    put("elderly", new AgeRange(60, MAX_AGE));
    put("elder", get("elderly"));
    put("young", new AgeRange(0, 29));
  }};

  private final Map<String, Integer> LOW_NUMBER_WORDS = new HashMap<String, Integer>() {{
    put("one", 1);
    put("two", 2);
    put("three", 3);
    put("four", 4);
    put("five", 5);
    put("six", 6);
    put("seven", 7);
    put("eight", 8);
    put("nine", 9);
    put("ten", 10);
    put("eleven", 11);
    put("twelve", 12);
    put("thirteen", 13);
    put("fourteen", 14);
    put("fifteen", 15);
    put("sixteen", 16);
    put("seventeen", 17);
    put("eighteen", 18);
    put("nineteen", 19);
  }};

  private final static Map<String, Integer> TENS_NUMBER_WORDS = new HashMap<String, Integer>() {{
    put("twenty", 20);
    put("thirty", 30);
    put("forty", 40);
    put("fifty", 50);
    put("sixty", 60);
    put("seventy", 70);
    put("eighty", 80);
    put("ninety", 90);
  }};

  private final String AGE                        = "(\\b\\d+|(?:[, .-]|" + getSimplePattern("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "and") + ")+)";
  private final String AGE_PHRASE                 = getCapturePattern("years old", "years of age", "years");
  private final String LESS_THAN_PREFIX           = getCapturePattern("under", "under the age of", "less than");
  private final String LESS_THAN_EQ_PREFIX        = getCapturePattern("maximum of", "upper limit of", "no more than", "at most");
  private final String GREATER_THAN_PREFIX        = getCapturePattern("over", "over the age of", "greater than");
  private final String GREATER_THAN_EQ_PREFIX     = getCapturePattern("minimum of", "lower limit of", "no less than", "at least");
  private final String LESS_THAN_AGE_PREFIX       = getCapturePattern("younger than");
  private final String LESS_THAN_EQ_AGE_PREFIX    = getCapturePattern("no older than");
  private final String GREATER_THAN_AGE_PREFIX    = getCapturePattern("older than");
  private final String GREATER_THAN_EQ_AGE_PREFIX = getCapturePattern("no younger than");
  private final String LESS_THAN_EQ_AGE_SUFFIX    = getCapturePattern("or younger");
  private final String GREATER_THAN_EQ_AGE_SUFFIX = getCapturePattern("or older");

  private static String getCapturePattern(String... elements) {
    return _getPattern(elements, true);
  }

  private static String getSimplePattern(String... elements) {
    return _getPattern(elements, false);
  }

  private static String _getPattern(String[] elements, boolean capture) {
    StringBuilder pattern = new StringBuilder();

    if (capture) {
      pattern.append("(");
    }
    for (int i = 0; i < elements.length; i++) {
      if (i > 0) {
        pattern.append("|");
      }
      pattern.append("\\b").append(Pattern.quote(elements[i])).append("\\b");
    }
    if (capture) {
      pattern.append(")");
    }
    return pattern.toString();
  }

  @Override
  public void close() { /* Nothing to close */ }

  /* Age Phrases:
  no older than [age] [age_phrase]
   */
  private AgeRange extractAgeRange(CharSequence query, String PREFIX, String EQ_PREFIX, String AGE_PREFIX, String EQ_AGE_PREFIX, String EQ_SUFFIX, Direction dir) {
    String message = query.toString().toLowerCase();

    Pattern pattern;
    Matcher matcher;

    pattern = Pattern.compile(PREFIX + " " + AGE + " " + AGE_PHRASE);
    matcher = pattern.matcher(message);
    if (matcher.find()) {
      log.debug("Found AGE=\"{}\" with PREFIX=\"{}\" and SUFFIX=\"{}\" (AGE_PRE)", matcher.group(2), matcher.group(1), matcher.group(3));
      return new AgeRange(dir, convertAge(matcher.group(2)) + dir.shift());
    }

    pattern = Pattern.compile(EQ_PREFIX + " " + AGE + " " + AGE_PHRASE);
    matcher = pattern.matcher(message);
    if (matcher.find()) {
      log.debug("Found AGE=\"{}\" with PREFIX=\"{}\" and SUFFIX=\"{}\" (AGE_PRE)", matcher.group(2), matcher.group(1), matcher.group(3));
      return new AgeRange(dir, convertAge(matcher.group(2)));
    }

    pattern = Pattern.compile(AGE_PREFIX + " " + AGE);
    matcher = pattern.matcher(message);
    if (matcher.find()) {
      log.debug("Found AGE=\"{}\" with PREFIX=\"{}\" (AGE_RANGE_PRE)", matcher.group(2), matcher.group(1));
      return new AgeRange(dir, convertAge(matcher.group(2)) + dir.shift());
    }

    pattern = Pattern.compile(EQ_AGE_PREFIX + " " + AGE);
    matcher = pattern.matcher(message);
    if (matcher.find()) {
      log.debug("Found AGE=\"{}\" with PREFIX=\"{}\" (AGE_RANGE_PRE)", matcher.group(2), matcher.group(1));
      return new AgeRange(dir, convertAge(matcher.group(2)));
    }

    pattern = Pattern.compile(AGE + " " + AGE_PHRASE + "? ?" + EQ_SUFFIX);
    matcher = pattern.matcher(message);
    if (matcher.find()) {
      log.debug("Found AGE=\"{}\" with PHRASE=\"{}\" and SUFFIX=\"{}\" (AGE_RANGE_SUF)", matcher.group(1), matcher.group(2), matcher.group(3));
      return new AgeRange(dir, convertAge(matcher.group(1)));
    }

    return AgeRange.NONE;
  }

  private AgeRange extractAgeRange(CharSequence query) {
    Pattern pattern;
    Matcher matcher;

    String message = query.toString().toLowerCase();

    final String RANGE_PREFIX = getCapturePattern("aged", "ages of", "their", "ages");
    final String RANGE_INFIX = getCapturePattern("to", "and", "or");

    pattern = Pattern.compile(RANGE_PREFIX + " " + AGE + "s? " + RANGE_INFIX + " " + AGE + "s?");
    matcher = pattern.matcher(message);
    if (matcher.find()) {
      log.debug("Found AGE1={} to AGE2={} with PREFIX={} and INFIX={} (AGED x TO y: {})", matcher.group(2), matcher.group(4), matcher.group(1), matcher.group(3), matcher.group(0));
      return new AgeRange(convertAge(matcher.group(2)), convertAge(matcher.group(4)));
    }

    for (String KNOWN_AGE_RANGE : KNOWN_AGE_RANGES.keySet()) {
      pattern = Pattern.compile("\\b" + KNOWN_AGE_RANGE + "\\b");
      matcher = pattern.matcher(message);
      if (matcher.find()) {
        log.debug("Found known age range {} expanding to {}.", KNOWN_AGE_RANGE, KNOWN_AGE_RANGES.get(KNOWN_AGE_RANGE));
        return KNOWN_AGE_RANGES.get(KNOWN_AGE_RANGE);
      }
    }

    return AgeRange.NONE;
  }

  private int convertAge(String phrase) {
    String[] tokens = phrase.split("\\s|-");

    int age = 0;
    try {
      age = Integer.parseInt(phrase);
      return age;
    } catch (NumberFormatException ex) {
      for (String token : tokens) {
        if (LOW_NUMBER_WORDS.containsKey(token)) {
          age += LOW_NUMBER_WORDS.get(token);
          continue;
        }

        if (TENS_NUMBER_WORDS.containsKey(token)) {
          age += TENS_NUMBER_WORDS.get(token);
          continue;
        }

        if (token.equals("hundred")) {
          age *= 100;
        }
      }
      return age;
    }
  }

  public AgeRange extract(CharSequence query) {
    AgeRange lower = extractAgeRange(query, LESS_THAN_PREFIX, LESS_THAN_EQ_PREFIX, LESS_THAN_AGE_PREFIX, LESS_THAN_EQ_AGE_PREFIX, LESS_THAN_EQ_AGE_SUFFIX, Direction.LOWER);
    AgeRange upper = extractAgeRange(query, GREATER_THAN_PREFIX, GREATER_THAN_EQ_PREFIX, GREATER_THAN_AGE_PREFIX, GREATER_THAN_EQ_AGE_PREFIX, GREATER_THAN_EQ_AGE_SUFFIX, Direction.UPPER);
    AgeRange other = extractAgeRange(query);

    AgeRange result = AgeRange.NONE;

    if (!(lower != AgeRange.NONE ^ upper != AgeRange.NONE ^ other != AgeRange.NONE) && lower != AgeRange.NONE) {
      log.error("Parsing conflict with message \"{}\" evaluates to LOW={} UP={} OTH={}", query, lower, upper, other);
      return null;
    } else if (lower != AgeRange.NONE) {
      result = lower;
    } else if (upper != AgeRange.NONE) {
      result = upper;
    } else if (other != AgeRange.NONE) {
      result = other;
    }

    if (result != AgeRange.NONE) {
      log.trace("Found age range {}", result);
    }
    return result;
  }

  @Override
  public AgeRange apply(CharSequence charSequence) {
    return extract(charSequence);
  }

  public static enum Direction {

    LOWER(MIN_AGE),
    UPPER(MAX_AGE);
    private final int value;

    private Direction(int value) {
      this.value = value;
    }

    public int getValue() {
      return this.value;
    }

    public int shift() {
      switch (this) {
        case LOWER:
          return -1;
        case UPPER:
          return 1;
        default:
          return 0;
      }
    }
  }

  public static class AgeRange implements Serializable {

    public static final AgeRange NONE = new AgeRange(Integer.MIN_VALUE, Integer.MAX_VALUE) {
      @Override public String toString() {
        return "AgeRange.NONE";
      }
    };

    public static final int LOWER_BOUND = 3;
    public static final int UPPER_BOUND = 94;

    int start;
    int end;

    public AgeRange(int start, int end) {
      this.start = Math.min(start, end);
      this.end = Math.max(start, end);
    }

    public AgeRange(Direction dir, int value) {
      switch (dir) {
        case LOWER:
          this.start = dir.getValue();
          this.end = Math.min(value, MAX_AGE);
          break;
        case UPPER:
          this.start = Math.max(value, MIN_AGE);
          this.end = dir.getValue();
          break;
      }
    }

    public static String format(int value) {
      assert value >= MIN_AGE;
      assert value <= MAX_AGE;
      if (value <= 12) {
        return "** AGE [ birth-12 ]";
      } else if (value < 20) {
        return "** AGE [ in teens ]";
      } else if (value >= 90) {
        return "** AGE [ 90 + ]";
      } else {
        return "** AGE [ in " + (value / 10 * 10) + "s ]";
      }
    }

    public int getEnd() {
      return end;
    }

    public int getStart() {
      return start;
    }

    @Override
    public String toString() {
      return "[" + start + ", " + end + "]";
    }
  }
}
