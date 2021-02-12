
"\nScript to generate contributor and pull request lists\n\nThis script generates contributor and pull request lists for release\nannouncements using Github v3 protocol. Use requires an authentication token in\norder to have sufficient bandwidth, you can get one following the directions at\n`<https://help.github.com/articles/creating-an-access-token-for-command-line-use/>_\nDon't add any scope, as the default is read access to public information. The\ntoken may be stored in an environment variable as you only get one chance to\nsee it.\n\nUsage::\n\n    $ ./scripts/announce.py <token> <revision range>\n\nThe output is utf8 rst.\n\nDependencies\n------------\n\n- gitpython\n- pygithub\n\nSome code was copied from scipy `tools/gh_lists.py` and `tools/authors.py`.\n\nExamples\n--------\n\nFrom the bash command line with $GITHUB token.\n\n    $ ./scripts/announce.py $GITHUB v1.11.0..v1.11.1 > announce.rst\n\n"
import codecs
import os
import re
import textwrap
from git import Repo
UTF8Writer = codecs.getwriter('utf8')
this_repo = Repo(os.path.join(os.path.dirname(__file__), '..', '..'))
author_msg = 'A total of %d people contributed patches to this release.  People with a\n"+" by their names contributed a patch for the first time.\n'
pull_request_msg = 'A total of %d pull requests were merged for this release.\n'

def get_authors(revision_range):
    pat = '^.*\\t(.*)$'
    (lst_release, cur_release) = [r.strip() for r in revision_range.split('..')]
    if ('|' in cur_release):
        (maybe_tag, head) = cur_release.split('|')
        assert (head == 'HEAD')
        if (maybe_tag in this_repo.tags):
            cur_release = maybe_tag
        else:
            cur_release = head
        revision_range = f'{lst_release}..{cur_release}'
    xpr = re.compile('Co-authored-by: (?P<name>[^<]+) ')
    cur = set(xpr.findall(this_repo.git.log('--grep=Co-authored', '--pretty=%b', revision_range)))
    cur |= set(re.findall(pat, this_repo.git.shortlog('-s', revision_range), re.M))
    pre = set(xpr.findall(this_repo.git.log('--grep=Co-authored', '--pretty=%b', lst_release)))
    pre |= set(re.findall(pat, this_repo.git.shortlog('-s', lst_release), re.M))
    cur.discard('Homu')
    pre.discard('Homu')
    authors = ([(s + ' +') for s in (cur - pre)] + [s for s in (cur & pre)])
    authors.sort()
    return authors

def get_pull_requests(repo, revision_range):
    prnums = []
    merges = this_repo.git.log('--oneline', '--merges', revision_range)
    issues = re.findall('Merge pull request \\#(\\d*)', merges)
    prnums.extend((int(s) for s in issues))
    issues = re.findall('Auto merge of \\#(\\d*)', merges)
    prnums.extend((int(s) for s in issues))
    commits = this_repo.git.log('--oneline', '--no-merges', '--first-parent', revision_range)
    issues = re.findall('^.*\\(\\#(\\d+)\\)$', commits, re.M)
    prnums.extend((int(s) for s in issues))
    prnums.sort()
    prs = [repo.get_pull(n) for n in prnums]
    return prs

def build_components(revision_range, heading='Contributors'):
    (lst_release, cur_release) = [r.strip() for r in revision_range.split('..')]
    authors = get_authors(revision_range)
    return {'heading': heading, 'author_message': (author_msg % len(authors)), 'authors': authors}

def build_string(revision_range, heading='Contributors'):
    components = build_components(revision_range, heading=heading)
    components['uline'] = ('=' * len(components['heading']))
    components['authors'] = ('* ' + '\n* '.join(components['authors']))
    tpl = textwrap.dedent('    {heading}\n    {uline}\n\n    {author_message}\n    {authors}').format(**components)
    return tpl

def main(revision_range):
    text = build_string(revision_range)
    print(text)
if (__name__ == '__main__'):
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Generate author lists for release')
    parser.add_argument('revision_range', help='<revision>..<revision>')
    args = parser.parse_args()
    main(args.revision_range)
