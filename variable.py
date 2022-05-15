import pathlib
from Preprocess import name_features


def get_project():
    with open(str(pathlib.Path(__file__).parent.resolve()) + "/name_project.txt", "r") as f:
        return f.read()


def update():
    with open(str(pathlib.Path(__file__).parent.resolve()) + "/name_project.txt", "r") as f:
        project = f.read()
    return "Data/" + project, project, 'apache/' + project, "../Data/" + project, "Data/" + project + "/blame/"


def get_name_github():
    with open(str(pathlib.Path(__file__).parent.resolve()) + "/name_project.txt", "r") as f:
        project = f.read()
    return project


def get_repo_full_name():
    with open(str(pathlib.Path(__file__).parent.resolve()) + "/name_project.txt", "r") as f:
        project = f.read()
    return 'apache/' + project


def get_name_dit_blame():
    with open(str(pathlib.Path(__file__).parent.resolve()) + "/name_project.txt", "r") as f:
        project = f.read()
    return "Data/" + project + "/blame/"


def get_key_issue():
    with open(str(pathlib.Path(__file__).parent.resolve()) + "/name_project.txt", "r") as f:
        project = f.read()
    if project == 'commons-math':
        key_issue = "MATH"
    elif project == 'cayenne':
        key_issue = "CAY"
    elif project == 'kylin':
        key_issue = "KYLIN"
    elif project == 'mahout':
        key_issue = "MAHOUT"
    elif project == 'jspwiki':
        key_issue = "JSPWIKI"
    elif project == 'commons-collections':
        key_issue = 'COLLECTIONS'
    elif project == 'manifoldcf':
        key_issue = 'CONNECTORS'
    elif project == 'commons-lang':
        key_issue = 'LANG'
    elif project == 'tika':
        key_issue = 'TIKA'
    elif project == 'kafka':
        key_issue = 'KAFKA'
    elif project == 'zookeeper':
        key_issue = 'ZOOKEEPER'
    elif project == 'zeppelin':
        key_issue = 'ZEPPELIN'
    elif project == 'shiro':
        key_issue = 'SHIRO'
    elif project == 'logging-log4j2':
        key_issue = 'LOG4J2'
    elif project == 'activemq-artemis':
        key_issue = 'ARTEMIS'
    elif project == 'openwebbeans':
        key_issue = 'OWB'
    elif project == 'shindig':
        key_issue = 'SHINDIG'
    elif project == 'directory-studio':
        key_issue = 'DIRSTUDIO'
    elif project == 'tapestry-5':
        key_issue = 'TAPESTRY'
    elif project == 'openjpa':
        key_issue = 'OPENJPA'
    elif project == 'knox':
        key_issue = 'KNOX'
    elif project == 'commons-configuration':
        key_issue = 'CONFIGURATION'
    elif project == 'xmlgraphics-batik':
        key_issue = 'XGC'
    elif project == 'deltaspike':
        key_issue = 'DELTASPIKE'
    return key_issue


project = get_project()
features_check_before_pre_process = name_features.JAVADIFF_FEATURES_DIFF + name_features.JAVADIFF_FEATURES_STATEMENT + \
                                    name_features.JAVADIFF_FEATURES_AST + name_features.STATIC_FEATURES
