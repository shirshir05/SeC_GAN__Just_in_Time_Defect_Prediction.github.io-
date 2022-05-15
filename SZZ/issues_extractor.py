import jira
import git
import time
import re
import json
from datetime import datetime
import variable
from SZZ.pydriller_szz import pydriller_szz
from functools import reduce


class Issue(object):
    def __init__(self, issue_id, type, priority, resolution, url, creation_time):
        self.issue_id = issue_id
        self.type = type
        self.priority = priority
        self.resolution = resolution
        self.url = url
        self.creation_time = creation_time

    def to_saveable_dict(self):
        return {'issue_id': self.issue_id, 'type': self.type, 'priority': self.priority, 'resolution': self.resolution,
                'url': self.url, 'creation_time': self.creation_time}

    def to_features_dict(self):
        return {'issue_id': self.issue_id, 'type': self.type, 'priority': self.priority, 'resolution': self.resolution}


class JiraIssue(Issue):
    def __init__(self, issue, base_url):
        super().__init__(issue.key.strip().split('-')[1], issue.fields.issuetype.name.lower(),
                         JiraIssue.get_name_or_default(issue.fields.priority, 'minor'),
                         JiraIssue.get_name_or_default(issue.fields.resolution, 'resolved'), base_url,
                         datetime.strptime(issue.fields.created, "%Y-%m-%dT%H:%M:%S.%f%z"))
        self.fields = {}
        for k, v in dict(issue.fields.__dict__).items():
            if k.startswith("customfield_") or k.startswith("__"):
                continue
            if type(v) in [str, type(None), type(0), type(0.1)]:
                self.fields[k] = str(v)
            elif hasattr(v, 'name'):
                self.fields[k] = v.name.replace('\n', '').replace(';', '.,')
            elif type(v) in [list, tuple]:
                lst = []
                for item in v:
                    if type(item) in [str]:
                        lst.append(item)
                    elif hasattr(item, 'name'):
                        lst.append(item.name)
                self.fields[k] = "@@@".join(lst)
        for k in self.fields:
            self.fields[k] = ' '.join(self.fields[k].split())

    @staticmethod
    def get_name_or_default(val, default):
        if val:
            return val.name.lower()
        return default


def get_jira_issues(project_name, url="http://issues.apache.org/jira", bunch=100):
    """
    A function that returns a list of all the issue in Jira.
    :param project_name: The key issue of the project as it appears in Jira(https://issues.apache.org/jira/secure/Dashboard.jspa)
    :type project_name: str
    :param url: URL to the site where the issue is located (deftly: http://issues.apache.org/jira)
    :type url: str
    :param bunch: (deftly: 100)
    :type bunch: int
    :return: issue -
    :rtype: list[JiraIssue]
    """
    jira_conn = jira.JIRA(url)
    all_issues = []
    extracted_issues = 0
    sleep_time = 30
    while True:
        try:
            issues = jira_conn.search_issues("project={0}".format(project_name), maxResults=bunch,
                                             startAt=extracted_issues)
            all_issues.extend(issues)
            extracted_issues = extracted_issues + bunch
            if len(issues) < bunch:
                break
        except Exception as e:
            sleep_time = sleep_time * 2
            if sleep_time >= 480:
                raise e
            time.sleep(sleep_time)
    print(f"Number issue {len(all_issues)}")
    return list(map(lambda issue: JiraIssue(issue, url), all_issues))


def _clean_commit_message(commit_message):
    if "git-svn-id" in commit_message:
        return commit_message.split("git-svn-id")[0]
    return ' '.join(commit_message.split())


def fix_renamed_files(files):
    """
    fix the paths of renamed files.
    before : u'tika-core/src/test/resources/{org/apache/tika/fork => test-documents}/embedded_with_npe.xml'
    after:
    u'tika-core/src/test/resources/org/apache/tika/fork/embedded_with_npe.xml'
    u'tika-core/src/test/resources/test-documents/embedded_with_npe.xml'
    :param files: self._files
    :return: list of modified files in commit
    """
    new_files = []
    for file in files:
        if "=>" in file:
            if "{" and "}" in file:
                # file moved
                src, dst = file.split("{")[1].split("}")[0].split("=>")
                fix = lambda repl: re.sub(r"{[\.a-zA-Z_/\-0-9]* => [\.a-zA-Z_/\-0-9]*}", repl.strip(), file)
                new_files.extend(map(fix, [src, dst]))
            else:
                # full path changed
                new_files.extend(map(lambda x: x.strip(), file.split("=>")))
                pass
        else:
            new_files.append(file)
    return new_files


class CommittedFile(object):
    """
    A class that represents modification (commit + file).
    """
    def __init__(self, sha, name, insertions, deletions):
        """
        A constructor who initializes the fields.
        :param sha: sha commit
        :type sha:  str
        :param name: name file
        :type name:  str
        :param insertions: The number of insertions that the commit made in the file.
        :type insertions:  str
        :param deletions: The number of deletions that the commit made in the file.
        :type deletions:  str
        """
        self.sha = sha
        self.name = fix_renamed_files([name])[0]
        if insertions.isnumeric():
            self.insertions = int(insertions)
            self.deletions = int(deletions)
        else:
            self.insertions = 0
            self.deletions = 0
        self.is_java = self.name.endswith(".java")
        self.is_test = 'test' in self.name


class CommittedModeFile(object):
    """
    A class that represents modification (commit + file) and the mode of the file.
    """
    def __init__(self, sha, name, mode):
        """
        A constructor who initializes the field fields.
        :param sha: sha commit
        :type sha:  str
        :param name: name file
        :type name:  str
        :param mode: rename (R), added (A), deleted(D) or modify(M)
        :type mode:  str
        """
        self.sha = sha
        self.name = fix_renamed_files([name])[0]
        self.mode = mode
        self.is_java = self.name.endswith(".java")
        self.is_test = 'test' in self.name


def _get_commits_files(repo):
    """
    The function returns all the commit for the project. The object returned is a dictionary that contains the sha of
    the commit as the key and the value is the CommittedFile type.
    CommittedFile contain information on the commit such as a number of insertions, deletions and etc.
    :param repo: Represents a git repository and
    allows you to query references, gather commit information, generate diffs, create and clone repositories query
    the log. :type repo: git.repo.base.Repo
    :rtype: dict[commit_sha, CommittedFile]
    """
    data = repo.git.log('--numstat', '--pretty=format:"sha: %H"').split("sha: ")
    comms = {}
    for d in data[1:]:
        d = d.replace('"', '').replace('\n\n', '\n').split('\n')
        commit_sha = d[0]
        comms[commit_sha] = []
        for x in d[1:-1]:
            insertions, deletions, name = x.split('\t')
            names = fix_renamed_files([name])
            comms[commit_sha].extend(list(map(lambda n: CommittedFile(commit_sha, n, insertions, deletions), names)))
    return dict(map(lambda x: (x, comms[x]), filter(lambda x: comms[x], comms)))


def _get_commits_modification_files(repo):
    """
    For each modification (commit + file) returns the mode (rename (R), added (A), deleted(D) or modify(M)) performed
    on the file.
    :param repo: Represents a git repository and allows you to query references, gather commit
    information, generate diffs, create and clone repositories query the log.
    :type repo: git.repo.base.Repo
    :rtype: dict[commit_sha, CommittedModeFile]
    """
    data = repo.git.log('--name-status', '--pretty=format:"sha: %H"').split("sha: ")
    comms = {}
    for d in data[1:]:
        d = d.replace('"', '').replace('\n\n', '\n').split('\n')
        commit_sha = d[0]
        comms[commit_sha] = []
        for x in d[1:-1]:
            try:
                mode, name = x.split('\t')
            except:
                print(x)
                continue
                pass
            names = fix_renamed_files([name])
            comms[commit_sha].extend(list(map(lambda n: CommittedModeFile(commit_sha, n, mode), names)))
    return dict(map(lambda x: (x, comms[x]), filter(lambda x: comms[x], comms)))


class Commit(object):
    """
    A class that represents commit.
    """
    def __init__(self, bug_id, git_commit, issue=None, files=None, is_java_commit=True):
        """
        A constructor who initializes the field fields.
        :param bug_id:
        :param git_commit:
        :param issue:
        :param files:
        :param is_java_commit:
        """
        self._commit_id = git_commit.hexsha
        self._repo_dir = git_commit.repo.working_dir
        self._issue_id = bug_id
        if files:
            self._files = files
        else:
            self._files = list(
                map(lambda f: CommittedFile(self._commit_id, f, '0', '0'), git_commit.stats.files.keys()))
        self._methods = list()
        self._commit_date = time.mktime(git_commit.committed_datetime.timetuple())
        self._commit_formatted_date = datetime.utcfromtimestamp(self._commit_date).strftime('%Y-%m-%d %H:%M:%S')
        self.issue = issue
        if issue:
            self.issue_type = self.issue.type
        else:
            self.issue_type = ''
        self.is_java_commit = is_java_commit
        self.is_all_tests = all(list(map(lambda x: not x.is_test, self._files)))

    @classmethod
    def init_commit_by_git_commit(cls, git_commit, bug_id='0', issue=None, files=None, is_java_commit=True):
        return Commit(bug_id, git_commit, issue, files=files, is_java_commit=is_java_commit)


def _commits_and_issues(repo, jira_issues):
    issues = dict(map(lambda x: (x.issue_id, x), jira_issues))
    issues_dates = sorted(list(map(lambda x: (x, issues[x].creation_time), issues)), key=lambda x: x[1], reverse=True)

    def replace(chars_to_replace, replacement, s):
        temp_s = s
        for c in chars_to_replace:
            temp_s = temp_s.replace(c, replacement)
        return temp_s

    def get_bug_num_from_comit_text(commit_text, issues_ids):
        text = replace("[]?#,:(){}'\"", "", commit_text.lower())
        text = replace("-_.=", " ", text)
        text = text.replace('bug', '').replace('fix', '')
        for word in text.split():
            if word.isdigit():
                if word in issues_ids:
                    return word
        return "0"

    commits = []
    java_commits = _get_commits_files(repo)
    for commit_sha in java_commits:
        git_commit = repo.commit(commit_sha)
        bug_id = "0"
        if all(list(map(lambda x: not x.is_java, java_commits[commit_sha]))):
            commit = Commit.init_commit_by_git_commit(git_commit, bug_id, None, java_commits[commit_sha], False)
            commits.append(commit)
            continue
        try:
            commit_text = _clean_commit_message(git_commit.message)
        except Exception as e:
            continue
        ind = 0
        for ind, (issue_id, date) in enumerate(issues_dates):
            date_ = date
            if date_.tzinfo:
                date_ = date_.replace(tzinfo=None)
            if git_commit.committed_datetime.replace(tzinfo=None) > date_:
                break
        issues_dates = issues_dates[ind:]
        bug_id = get_bug_num_from_comit_text(commit_text, set(map(lambda x: x[0], issues_dates)))
        commits.append(
            Commit.init_commit_by_git_commit(git_commit, bug_id, issues.get(bug_id), java_commits[commit_sha]))
    return commits


def extract_json(repo_path, jira_key, repo_full_name, out_json, out_non_tests_json):
    """
    Create file with fix_commit_hash and issue.

    :param repo_path: The path of the folder where the project is located. The folder must contain a git
    folder.
    :type repo_path: str
    :param jira_key: The key issue of the project as it appears in Jira(https://issues.apache.org/jira/secure/Dashboard.jspa)
    :type jira_key: str
    :param repo_full_name: The full name of the project listed on github (default: with apache)
    :type repo_full_name: str
    :param out_json: Name a folder for storing the files of the modification that induced defect
    :type out_json: str
    :param out_non_tests_json: Name a folder for storing the files of the modification that induced defect (without
    modification that change only test file).
    :type out_non_tests_json: str
    """
    issues = get_jira_issues(jira_key)
    commits = _commits_and_issues(git.Repo(repo_path), issues)
    # save_to_json(commits, repo_full_name, out_json)
    # save_to_json(list(filter(lambda x: not x.is_all_tests, commits)), repo_full_name, out_non_tests_json)
    to_many_files = list(filter(lambda x: len(x._files) < 6, commits))
    save_to_json(to_many_files, repo_full_name, out_json)
    save_to_json(list(filter(lambda x: not x.is_all_tests, to_many_files)), repo_full_name, out_non_tests_json)


def save_to_json(commits, repo_full_name, out_json):
    issued_ = list(filter(lambda c: c.issue is not None, commits))
    buggy = list(filter(lambda c: c.issue.type.lower() == 'bug', issued_))
    print(f"{len(buggy)} number defect issue for {out_json}")
    bugs_json = list(map(lambda c: {"repo_name": repo_full_name, 'fix_commit_hash': c._commit_id,
                                    "earliest_issue_date": c.issue.creation_time.strftime("%Y-%m-%dT%H:%M:%SZ")},
                         buggy))
    with open(out_json, 'w') as out:
        json.dump(bugs_json, out)


def merge_commit(all_commits, all_commits_mode, name_dir):
    """
    Create file modification_commit with this columns: 'commit_sha', 'file_name', 'mode', 'is_java', 'is_test','insertions', 'deletions'

    :param all_commits: The information is extracted from the function _get_commits_files
    :type all_commits: list[CommittedFile]
    :param all_commits_mode: The information is extracted from the function _get_commits_modification_files
    :type all_commits_mode: list[CommittedModeFile]
    :param name_dir: Name a folder for storing the files
    :type name_dir: str
    """

    def write_modification():
        import csv
        file = open(name_dir + '/modification_commit.csv', 'w', newline='')
        with file:
            write = csv.writer(file)
            write.writerows([['commit_sha', 'file_name', 'mode', 'is_java', 'is_test', 'insertions', 'deletions']])
            for commit in result.values():
                write.writerows([[commit[0].sha, commit[0].name, commit[1].mode, commit[0].is_java, commit[0].is_test,
                                  commit[0].insertions, commit[0].deletions]])

    dic1 = dict(map(lambda x: ((x.sha, x.name), x), all_commits))
    dic2 = dict(map(lambda x: ((x.sha, x.name), x), all_commits_mode))
    result = {}
    for key in (dic1.keys() | dic2.keys()):
        if key in dic1: result.setdefault(key, []).append(dic1[key])
        if key in dic2: result.setdefault(key, []).append(dic2[key])
    write_modification()


def main_szz(location_github_project, name_github, key_issue, repo_full_name, name_dir):
    """
    The main function that runs the SZZ algorithm.
    At the end of the function, a file is written in the folder "name_dir" that contains the modification that induced
    defect.

    :param location_github_project: The path of the folder where the project is located. The folder must contain a git
    folder.
    :type location_github_project: str
    :param name_github: Project name as it appears on github
    :type name_github: str
    :param key_issue: The key issue of the project as it appears in Jira(https://issues.apache.org/jira/secure/Dashboard.jspa)
    :type key_issue: str
    :param repo_full_name: The full name of the project listed on github (default: with apache)
    :type repo_full_name: str
    :param name_dir: Name a folder for storing the files of the modification that induced defect
    :type name_dir: str
    """
    java_commits_mode = _get_commits_modification_files(git.Repo(location_github_project))
    java_commits = _get_commits_files(git.Repo(location_github_project))

    all_commits = reduce(list.__add__, list(java_commits.values()), [])
    all_commits_mode = reduce(list.__add__, list(java_commits_mode.values()), [])
    merge_commit(all_commits, all_commits_mode, name_dir)

    extract_json(location_github_project, key_issue, repo_full_name, name_dir + '/bugfixes.json',
                 name_dir + 'non_tests_bugfixes.json')

    # Create file 'bugfix_commit', 'filename', 'bic'
    pydriller_szz(location_github_project, name_dir + '/bugfixes.json',
                  name_dir + r"/pydriller_" + name_github + "_bugfixes_bic")
