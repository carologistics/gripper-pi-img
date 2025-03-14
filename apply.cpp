#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <pwd.h>
#include <grp.h>
#include <unistd.h>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <acl/libacl.h>
#include <sys/acl.h>

inline bool match_perms(const mode_t target, acl_entry_t entry) {
    bool diff = false;

    acl_permset_t permset;
    acl_get_permset(entry, &permset);
    if (acl_get_perm(permset, ACL_READ) != (target & S_IRGRP)) {
      diff = true;
      if(target & S_IRGRP) {
        acl_add_perm(permset, ACL_READ);
      } else {
        acl_delete_perm(permset, ACL_READ);
      }
    }
    if (acl_get_perm(permset, ACL_WRITE) != (target & S_IWGRP)) {
      diff = true;
      if(target & S_IWGRP) {
        acl_add_perm(permset, ACL_WRITE);
      } else {
        acl_delete_perm(permset, ACL_WRITE);
      }
    }
    if (acl_get_perm(permset, ACL_EXECUTE) != (target & S_IXGRP)) {
      diff = true;
      if(target & S_IXGRP) {
        acl_add_perm(permset, ACL_EXECUTE);
      } else {
        acl_delete_perm(permset, ACL_EXECUTE);
      }
    }
    if(diff) {
      acl_set_permset(entry, permset);
    }
    return diff;
}

inline bool set_permset(const mode_t target, acl_entry_t entry) {
  bool diff = false;
  acl_permset_t permset;

  if (acl_get_permset(entry, &permset) == -1){
    std::cerr << "Failed to get permset" << std::endl;
    return false;
  }

  if(target & S_IRGRP) {
    diff = true;
    acl_add_perm(permset, ACL_READ);
  }
  if(target & S_IWGRP) {
    diff = true;
    acl_add_perm(permset, ACL_WRITE);
  }
  if(target & S_IXGRP) {
    diff = true;
    acl_add_perm(permset, ACL_EXECUTE);
  }
  acl_set_permset(entry, permset);
  return diff;
}

bool set_permset_for_uid(const std::string path, const int sudo_uid, const mode_t mode) {
    acl_t acl = acl_get_file(path.c_str(), ACL_TYPE_ACCESS);
    if (acl == nullptr) {
        std::cerr << "Failed to get ACL from file: " << path << std::endl;
        return false;
    }

    acl_entry_t entry;
    int entryId = ACL_FIRST_ENTRY;
    bool found = false;
    bool modified = false;

    while (acl_get_entry(acl, entryId, &entry) == 1) {
        entryId = ACL_NEXT_ENTRY;

        acl_tag_t tagType;
        acl_get_tag_type(entry, &tagType);

        if (tagType == ACL_USER) {
            uid_t* entryUid = static_cast<uid_t*>(acl_get_qualifier(entry));
            if (*entryUid == sudo_uid) {
                found = true;
                modified = match_perms(mode, entry);
                acl_free(entryUid);
                break;
            }
            acl_free(entryUid);
        }
    }

    if(acl_valid(acl) == -1) {
      std::cerr << "PRE_Invalid ACL" << std::endl;
      acl_free(acl);
      return false;
    }
    if (!found) {
      acl_entry_t entry;
      if(acl_create_entry(&acl, &entry) == -1) {
        std::cerr << "Failed to create ACL entry for user: " << sudo_uid << std::endl;
        acl_free(acl);
        return false;
      }
      acl_set_tag_type(entry, ACL_USER);
      id_t id = sudo_uid;
      acl_set_qualifier(entry, &id);

      acl_permset_t permset;

      modified = set_permset(mode, entry);
    }

    if(modified) {
      acl_calc_mask(&acl);
      if(acl_valid(acl) == -1) {
        std::cerr << "Invalid ACL" << std::endl;
        acl_free(acl);
        return false;
      }
      if(acl_set_file(path.c_str(), ACL_TYPE_ACCESS, acl) == -1) {
        std::cerr << "Error setting ACL on " << path << ": "
                  << strerror(errno) << std::endl;
        acl_free(acl);
        return false;
      }
    }
    acl_free(acl);
    return true;
}

int main() {
  struct stat buf;
  std::string line;
  uid_t uid;
  gid_t gid;
  mode_t mode;
  size_t pos1, pos2, pos3, start;

  int sudo_uid = std::stoi(getenv("SUDO_UID"));

  while(std::cin) {
      getline(std::cin, line);
      if(line.empty()) {
          break;
      }

      pos1 = line.find(' ');
      mode = static_cast<mode_t>(std::stoul(line.substr(0, pos1),
                                            nullptr, 8));

      start = pos1 + 1;
      pos2 = line.find(' ', start);
      uid = stoi(line.substr(start, pos2 - start));

      // Find the position of the third space starting from the character after pos2
      start = pos2 + 1;
      pos3 = line.find(' ', start);
      gid = std::stoi(line.substr(start, pos3 - start));

      // Extract everything after the third space
      std::string file = line.substr(pos3 + 1);
      if(lchown(file.c_str(), uid, gid) == -1) {
          std::cerr << "Error changing owner/group on " << file << std::endl;
      }
      if(lstat(file.c_str(), &buf) == -1) {
          std::cerr << "Error getting file status on " << file << ": "
                    << strerror(errno) << std::endl;
      } else {
        if(S_ISLNK(buf.st_mode)) {
          continue;
        }
      }
      if(chmod(file.c_str(), mode) == -1) {
          std::cerr << "Error changing file permissions on "
                    << file << ": " << strerror(errno) << std::endl;
      }
      if(!set_permset_for_uid(file, sudo_uid, mode)) {
          std::cerr << "Error setting ACL for user " << sudo_uid
                    << " on " << file << std::endl;
      }
  }
}
