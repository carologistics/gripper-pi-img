��    -      �  =   �      �     �     �          +     G     d  1   �     �  ,   �  m   �  #   ^  (   �     �     �     �     �  %     '   )     Q     `  #   r  "   �     �     �     �     �       #   -     Q  "   g  5   �  !   �     �     �            "   =     `  &  x     �     �     �  !   �  #   �  �    B        I  3   Y  F   �  )   �  1   �  \   0  C   �  w   �  �   I  =     U   K     �  :   �  i   �  5   K  v   �  O   �  8   H  '   �  Z   �  J     G   O  9   �  )   �  8   �  K   4  J   �  "   �  l   �  J   [  �   �  :   :  0   u  '   �  9   �  ;     4   D    y     �'  C   �'  >   �'  >   (  <   G(                                                      ,   -             &      )   *                       
                       "   	       '                  %      #          (   !          +             $          %s maintainer is %s not %s
  DeLink %s [%s]
  DeLink limit of %sB hit.
 *** Failed to link %s to %s Archive had no package field Archive has no control record Cannot get debconf version. Is debconf installed? Compress child Compressed output %s needs a compression set DB format is invalid. If you upgraded from an older version of apt, please remove and re-create the database. DB is old, attempting to upgrade %s DB was corrupted, file renamed to %s.old E:  E: Errors apply to file  Error processing contents %s Error processing directory %s Error writing header to contents file Failed to create IPC pipe to subprocess Failed to fork Failed to open %s Failed to read the override file %s Failed to read while computing MD5 Failed to readlink %s Failed to rename %s to %s Failed to resolve %s Failed to stat %s IO to subprocess/file failed Internal error, failed to create %s No selections matched Package extension list is too long Some files are missing in the package file group `%s' Source extension list is too long Tree walking failed Unable to get a cursor Unable to open %s Unable to open DB file %s: %s Unknown compression algorithm '%s' Unknown package record! Usage: apt-ftparchive [options] command
Commands: packages binarypath [overridefile [pathprefix]]
          sources srcpath [overridefile [pathprefix]]
          contents path
          release path
          generate config [groups]
          clean config

apt-ftparchive generates index files for Debian archives. It supports
many styles of generation from fully automated to functional replacements
for dpkg-scanpackages and dpkg-scansources

apt-ftparchive generates Package files from a tree of .debs. The
Package file contains the contents of all the control fields from
each package as well as the MD5 hash and filesize. An override file
is supported to force the value of Priority and Section.

Similarly apt-ftparchive generates Sources files from a tree of .dscs.
The --source-override option can be used to specify a src override file

The 'packages' and 'sources' command should be run in the root of the
tree. BinaryPath should point to the base of the recursive search and 
override file should contain the override flags. Pathprefix is
appended to the filename fields if present. Example usage from the 
Debian archive:
   apt-ftparchive packages dists/potato/main/binary-i386/ > \
               dists/potato/main/binary-i386/Packages

Options:
  -h    This help text
  --md5 Control MD5 generation
  -s=?  Source override file
  -q    Quiet
  -d=?  Select the optional caching database
  --no-delink Enable delinking debug mode
  --contents  Control contents file generation
  -c=?  Read this configuration file
  -o=?  Set an arbitrary configuration option W:  W: Unable to read directory %s
 W: Unable to stat %s
 Waited for %s but it wasn't there realloc - Failed to allocate memory Project-Id-Version: apt 1.0.5
Report-Msgid-Bugs-To: APT Development Team <deity@lists.debian.org>
PO-Revision-Date: 2012-09-25 20:19+0300
Last-Translator: A. Bondarenko <artem.brz@gmail.com>
Language-Team: Українська <uk@li.org>
Language: uk
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
X-Generator: KBabel 1.11.1
Plural-Forms: nplurals=3; plural=(n%10==1 && n%100!=11 ? 0 : n%10>=2 && n%10<=4 && (n%100<10 || n%100>=20) ? 1 : 2);
   пакунок %s супроводжується %s, а не %s
 DeLink %s [%s]
  Перевищено ліміт в %sB в DeLink.
 *** Не вдалося створити посилання %s на %s Архів не мав поля 'package' В архіві немає запису 'control' Неможливо визначити версію debconf. Він встановлений? Процес-нащадок, що виконує пакування Для отримання стиснутого виводу %s необхідно ввімкнути стиснення Невірний формат БД. Якщо ви оновилися зі старої версії apt, будь-ласка видаліть і наново створіть базу-даних. БД застаріла, намагаюсь оновити %s БД була пошкоджена, файл перейменований на %s.old П:  П: Помилки відносяться до файлу  Помилка обробки повного переліку вмісту пакунків (Contents) %s Помилка обробки директорії %s Помилка запису заголовка в повний перелік вмісту пакунків (Contents) Не вдалося створити IPC канал для підпроцесу Не вдалося породити процес (fork) Не вдалося відкрити %s Не вдалося прочитати файл перепризначень (override) %s Помилка зчитування під час обчислення MD5 Не вдалося прочитати посилання (readlink) %s Не вдалося перейменувати %s на %s Не вдалося визначити %s Не вдалося одержати атрибути %s Помилка уведення/виводу в підпроцес/файл Внутрішня помилка, не вдалося створити %s Збігів не виявлено Список розширень, припустимих для пакунків, занадто довгий У групі пакунків '%s' відсутні деякі файли Список розширень, припустимих для пакунків з вихідними текстами, занадто довгий Не вдалося зробити обхід дерева Неможливо одержати курсор Не вдалося відкрити %s Не вдалося відкрити файл БД %s: %s Невідомий алгоритм стиснення '%s' Невідомий запис про пакунок! Використання: apt-ftparchive [параметри] команда
Команди:  packages binarypath [overridefile [pathprefix]]
          sources srcpath [overridefile [pathprefix]]
          contents path
          release path
          generate config [groups]
          clean config

apt-ftparchive генерує індексні файли архівів Debian. Він підтримує
безліч стилів генерації: від повністю автоматичного до функціональної заміни
програм dpkg-scanpackages і dpkg-scansources

apt-ftparchive генерує файли Package (переліки пакунків) для дерева
тек, що містять файли .deb. Файл Package містить у собі керуючі
поля кожного пакунка, а також хеш MD5 і розмір файлу. Значення керуючих
полів "пріоритет" (Priority) і "секція" (Section) можуть бути змінені з
допомогою файлу override.

Крім того, apt-ftparchive може генерувати файли Sources з дерева
тек, що містять файли .dsc. Для вказівки файлу override у цьому 
режимі можна використати параметр --source-override.

Команди 'packages' і 'sources' треба виконувати, перебуваючи в кореневій теці
дерева, що ви хочете обробити. BinaryPath повинен вказувати на місце,
з якого починається рекурсивний обхід, а файл перепризначень (override)
повинен містити запис про перепризначення керуючих полів. Якщо був зазначений
Pathprefix, то його значення додається до керуючих полів, що містять
імена файлів. Приклад використання для архіву Debian:
   apt-ftparchive packages dists/potato/main/binary-i386/ > \
               dists/potato/main/binary-i386/Packages

Параметри:
  -h    Цей текст
  --md5 Керування генерацією MD5-хешів
  -s=?  Вказати файл перепризначень (override) для пакунків з вихідними текстами
  -q    Не виводити повідомлення в процесі роботи
  -d=?  Вказати кешуючу базу даних (не обов'язково)
  --no-delink Включити режим налагодження процесу видалення файлів
  --contents  Керування генерацією повного переліку вмісту пакунків
              (файлу Contents)
  -c=?  Використати зазначений конфігураційний файл
  -o=?  Вказати довільний параметр конфігурації У:  У: Не вдалося прочитати директорію %s
 У: Неможливо прочитати атрибути %s
 Очікував на %s, але його там не було realloc - Не вдалося виділити пам'ять 