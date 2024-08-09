from pysvn_tuto import get_svn_logs

if __name__ == "__main__":
    path = r"D:\dev\AutoPlanning\trunk\AP_trunk_pure\mod_APSurgicalGuide"
    logs = get_svn_logs(path)
    print(len(logs), end=" 개의 로그 \n")