package edu.utdallas.hltri.knowledge;

import com.google.common.base.Strings;
import edu.utdallas.hltri.conf.Config;
import edu.utdallas.hltri.io.ExternalBinarySearch;
import edu.utdallas.hltri.io.StusMagicLargeFileReader;
import edu.utdallas.hltri.logging.Logger;
import edu.utdallas.hltri.util.AbstractExpander;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Manages Wikipedia Redirects
 * @author travis
 */
@SuppressWarnings("WeakerAccess")
public class WikiRedirectManager extends AbstractExpander<CharSequence, String> implements Closeable {

  private static final Logger log = Logger.get(WikiRedirectManager.class);

  private final StusMagicLargeFileReader _article_id2name,
    _article_name2id,
    _redirect_id2name,
    _redirect_name2id;
  
  static final char SEPARATOR = '\t';


  public WikiRedirectManager(final String pageIds,
                         final String pageNames,
                         final String redirectIds,
                         final String redirectNames) {
    super("Wikipedia:Redirects");
    try {
      _article_id2name = new StusMagicLargeFileReader(new File(pageIds));
      _article_name2id = new StusMagicLargeFileReader(new File(pageNames));
      _redirect_id2name = new StusMagicLargeFileReader(new File(redirectIds));
      _redirect_name2id = new StusMagicLargeFileReader(new File(redirectNames));
    } catch (IOException ex) {
      throw new RuntimeException("Unable to initialize WikiDataManager", ex);
    }
  }

  public WikiRedirectManager() {
    super("Wikipedia:Redirects");
    final Config conf = Config.load("insight.wiki");
    
    final File _pageIds = conf.getFile("PAGE_IDS_PATH");
    final File _pageNames = conf.getFile("PAGE_NAMES_PATH");
    final File _redirectIds = conf.getFile("REDIRECT_IDS_PATH");
    final File _redirectNames = conf.getFile("REDIRECT_NAMES_PATH");

    assert (_pageIds.exists());
    assert (_pageNames.exists());
    assert (_redirectIds.exists());
    assert (_redirectNames.exists());

    try {
      _article_id2name = new StusMagicLargeFileReader(_pageIds);
      _article_name2id = new StusMagicLargeFileReader(_pageNames);
      _redirect_id2name = new StusMagicLargeFileReader(_redirectIds);
      _redirect_name2id = new StusMagicLargeFileReader(_redirectNames);
    } catch (IOException ex) {
      throw new RuntimeException("Unable to initialize WikiDataManager", ex);
    }
  }

  @Override
  public String getName() {
    return "Wikipedia:Redirects";
  }

  private List<String> getIdsFromName(String term, StusMagicLargeFileReader reader) {
    int delim1;
    int delim2;
    String name;
    String id;
    String namespace;
    List<String> ids = new ArrayList<>();

    for (String page : ExternalBinarySearch.binarySearch(reader, term + SEPARATOR)) {
      delim1 = page.indexOf('\t');
      delim2 = page.lastIndexOf('\t');
      name = page.substring(0, delim1);
      namespace = page.substring(delim1 + 1, delim2);
      id = page.substring(delim2 + 1);

      log.trace("found NAME={} ID={} NAMESPACE={}", name, id, namespace);
      if (!name.equals(term) || !namespace.equals("0")) {
        continue;
      }

      log.trace("added NAME={} ID={} NAMESPACE={}",name, id, namespace);
      ids.add(id);
    }

    return ids;
  }

  private String getNameFromId(String givenId, StusMagicLargeFileReader reader) {
    List<String> names = ExternalBinarySearch.binarySearch(reader, givenId + SEPARATOR);

    if (names.isEmpty()) {
      return null;
    }

    String line = names.get(0);

    int delim1 = line.indexOf('\t');
    int delim2 = line.lastIndexOf('\t');
    String id = line.substring(0, delim1);
    String namespace = line.substring(delim1 + 1, delim2);
    String name = line.substring(delim2 + 1);

    if (!id.equals(givenId) || !namespace.equals("0"))
      return null;

    log.trace("found NAME={} from ID={}", name, id);
    return name;
  }

  public String getRedirectNameFromRedirectId(String id) {
    log.trace("mapped ID={} to redirect '{}'", id, getNameFromId(id, _redirect_id2name));
    return getNameFromId(id, _redirect_id2name);
  }

  public String getPageNameFromPageId(String id) {
    log.trace("mapped NAME={} to ID={}", getNameFromId(id, _article_id2name), id);
    return getNameFromId(id, _article_id2name);
  }

  public String getPageIdFromPageName(String name) {
    List<String> ids = getIdsFromName(name, _article_name2id);

    if (ids.isEmpty()) {
      log.debug("Unable to resolve page id {}.", name);
      return null;
    }

    log.trace("mapped ID={} to page '{}'", ids.get(0), name);
    return ids.get(0);
  }

  public List<String> getRedirectIdsFromRedirectName(String name) {
    return getIdsFromName(name, _redirect_name2id);
  }

  public String getRedirectTarget(String pageName) {
    while (true) {
      String id = getPageIdFromPageName(pageName);

      if (id == null) {
        return null;
      }

      String redirectName = getRedirectNameFromRedirectId(id);
      log.trace("resolved NAME={} ID={} to REDIRECT={}.", pageName, id, redirectName);

      if (redirectName == null || redirectName.equals(pageName)) {
        return pageName;
      }

      pageName = redirectName;
    }
  }

  public List<String> getRedirects(String pageName) {
    pageName = format(pageName);

    if (pageName == null) {
      return Collections.emptyList();
    }

    List<String> pageNames = new ArrayList<>();

    log.trace("found IDS={} from PAGE='{}'.", getRedirectIdsFromRedirectName(pageName), pageName);
    for (String id : getRedirectIdsFromRedirectName(pageName)) {
      String name = getPageNameFromPageId(id);

      log.trace("found NAME='{}' from ID={}.", name, id);
      if (!Strings.isNullOrEmpty(name)) {
        pageNames.add(unformat(name));
      }
    }

    return pageNames;
  }

  @Override public List<String> getExpansions(CharSequence name) {
    String target = getRedirectTarget(format(name.toString()));
    if (target == null)
      return Collections.emptyList();
    return getRedirects(target);
  }

  @Override public void close() {
    try {
      _article_id2name.close();
      _article_name2id.close();
      _redirect_id2name.close();
      _redirect_name2id.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public String format(String phrase) {
    return phrase.replaceAll(" ", "_").toLowerCase();
  }

  @SuppressWarnings("WeakerAccess")
  public String unformat(String phrase) {
    return phrase.replaceAll("_", " ");
  }

  public static void main(String... args) {
    final String basePath = "/shared/aifiles/disk1/travis/data/ontologies/wikipedia/20100622/";
    final WikiRedirectManager wdm = new WikiRedirectManager(
        basePath + "lower_articles_id2name.tsv",
        basePath + "lower_articles_name2id.tsv",
        basePath + "lower_redirect_id2name.tsv",
        "basePath + \"lower_redirect_name2id.tsv");

    for (String arg : args) {
      for (String redirect : wdm.expand(arg)) {
        System.out.printf("  %s\n", redirect.replaceAll("_", " "));
      }
    }
  }
}
