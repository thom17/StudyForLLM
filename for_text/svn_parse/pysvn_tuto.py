import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime

import re

def get_svn_log(path):
    try:
        result = subprocess.run(['svn', 'log', path, '--xml'], capture_output=True, check=True)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error fetching SVN log: {e}")
        return None


def parse_svn_log(log_xml):
    root = ET.fromstring(log_xml)
    log_entries = []

    for logentry in root.findall('logentry'):
        entry = {
            'revision': logentry.attrib['revision'],
            'author': logentry.find('author').text,
            'date': datetime.strptime(logentry.find('date').text, '%Y-%m-%dT%H:%M:%S.%fZ'),
            'message': logentry.find('msg').text.strip() if logentry.find('msg') is not None and logentry.find(
                'msg').text is not None else '',
            'changed_paths': []
        }

        paths = logentry.find('paths')
        if paths is not None:
            for path in paths.findall('path'):
                entry['changed_paths'].append({
                    'action': path.attrib['action'],
                    'path': path.text
                })

        log_entries.append(entry)

    return log_entries


def get_svn_diff_files(path, revision):
    try:
        result = subprocess.run(['svn', 'diff', '--summarize', '-c', str(revision), path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error fetching SVN diff for revision {revision}: {e}")
        return None

def parse_svn_diff(diff_output):
    changed_files = []
    diff_lines = diff_output.splitlines()

    for line in diff_lines:
        match = re.match(r'^[A-Z]\s+(.*)', line)
        if match:
            action = line[0]
            file_path = match.group(1)
            changed_files.append({'action': action, 'path': file_path})

    return changed_files

def diff_main(path, revision):
    project_a_path = path
    revision_number = revision  # 원하는 리비전 번호로 변경

    diff_output = get_svn_diff_files(project_a_path, revision_number)
    if diff_output:
        changed_files = parse_svn_diff(diff_output)
        # print(f"Changed files in revision {revision_number}:")
        # for changed_file in changed_files:
        #     action = changed_file['action']
        #     action_desc = ''
        #     if action == 'A':
        #         action_desc = 'Added'
        #     elif action == 'D':
        #         action_desc = 'Deleted'
        #     elif action == 'M':
        #         action_desc = 'Modified'
        #     elif action == 'R':
        #         action_desc = 'Replaced'
        #     print(f"{action_desc}: {changed_file['path']}")
        #
        return changed_files




def get_svn_logs(project_a_path = r'D:\dev\AutoPlanning\trunk\AP_trunk_pure'):
    project_a_path = project_a_path

    file_path_map = {}

    log_entries = None
    log_xml = get_svn_log(project_a_path)
    if log_xml:
        log_entries = parse_svn_log(log_xml)

        # for entry in log_entries:
        #     print(f"Revision: {entry['revision']}")
        #     print(f"Author: {entry['author']}")
        #     print(f"Date: {entry['date']}")
        #     print(f"Message: {entry['message']}")
        #     print('Changed paths:')
        #     for changed_path in entry['changed_paths']:
        #         print(f"  {changed_path['path']} ({changed_path['action']})")
        #     print('-' * 40)

        print(len(log_entries), "개의 로그")

        for idx, entry in enumerate(log_entries):
            # print(f"Revision: {entry['revision']}" ,end=" : ")
            # # print(f"Author: {entry['author']}")
            # # print(f"Date: {entry['date']}")
            # # print(f"Message: {entry['message']}")
            # # print('Changed paths:')
            # # for changed_path in entry['changed_paths']:
            # #     print(f"  {changed_path['path']} ({changed_path['action']})")
            # # print('-' * 40)

            paths = diff_main(project_a_path, entry['revision'])
            entry['changed_paths'] = paths

            # print(f"{len(paths)} changes")
    return log_entries


if __name__ == "__main__":
    logs = get_svn_logs()
