��    `        �         (  l   )     �  �  �  b   n  M   �  H     p   h  �   �  q   {  �   �  �   �  �   �  �   �  9   $  #   ^     �     �     �  )   �  	   �  3         4  �   P      �  ,      $   -     R      g     �     �  #   �  !   �          !  <   :  <   w  %   �  %   �                :     P     g     v     �     �     �  �   �  &   �     �     �     �  �     d   �     U     l  $   �  u   �  C     =   b     �  &   �  +   �       (     )   F     p     �    �  (   �  /  �  �   "  }   �"  .   >#  F   m#  "   �#  -   �#     $  
   %$     0$  2   C$  $   v$  ,   �$  '   �$  '   �$     %     %  +   3%     _%     t%     �%     �%     �%     �%     �%  �  �%  w   �'     (  $   (  b   E,  U   �,  J   �,  x   I-  �   �-  }   q.  �   �.    �/  �   �0  �   �1  :   V2  $   �2     �2     �2  $   �2  1   3     D3  9   T3     �3  �   �3  ,   94  =   f4  .   �4  "   �4  (   �4  '   5  ,   G5  -   t5  '   �5     �5  #   �5  <   6  <   H6  %   �6  %   �6  '   �6     �6     7     47     K7     \7     u7     �7  #   �7  �   �7  +   �8  +   �8  !   �8  (   9  �   19  g   *:     �:     �:      �:  i   �:  D   T;  E   �;     �;  0   �;  7   )<     a<  2   s<  2   �<      �<  !   �<  y  =  +   �A  V  �A  �   D  �   �D  :   oE  W   �E  -   F  4   0F  !   eF  
   �F     �F  6   �F  $   �F  7   G  ,   9G  /   fG     �G     �G  0   �G     �G     �G     H     H     H     &H     6H     B   "   J       M   H   ?      
   2      K   G               %   [           O   (   9                 X   .      8       $                                   1   +   0          3             S       ,         *       =           E          U   <   '   /                         ^       4             C       P   @   ;          ]   T           )   W      	       R       _   7   -   &   N         Z   Y   `      >       A      D   #   5       V   L                F   I   6   Q   !   \   :    
  PID    start at this PID; default is 1 (init)
  USER   show only trees rooted at processes of this user

 
Display a tree of processes.

        killall -l, --list
       killall -V, --version

  -e,--exact          require exact match for very long names
  -I,--ignore-case    case insensitive process name match
  -g,--process-group  kill process group instead of process
  -y,--younger-than   kill processes younger than TIME
  -o,--older-than     kill processes older than TIME
  -i,--interactive    ask for confirmation before killing
  -l,--list           list all known signal names
  -q,--quiet          don't print complaints
  -r,--regexp         interpret NAME as an extended regular expression
  -s,--signal SIGNAL  send this signal instead of SIGTERM
  -u,--user USER      kill only process(es) running as USER
  -v,--verbose        report if the signal was successfully sent
  -V,--version        display version information
  -w,--wait           wait for processes to die
  -n,--ns PID         match processes that belong to the same namespaces
                      as PID
   -4,--ipv4             search IPv4 sockets only
  -6,--ipv6             search IPv6 sockets only
   -C, --color=TYPE    color process by attribute
                      (age)
   -Z, --security-context
                      show security attributes
   -Z,--context REGEXP kill only process(es) having context
                      (must precede other arguments)
   -a, --arguments     show command line arguments
  -A, --ascii         use ASCII line drawing characters
  -c, --compact-not   don't compact identical subtrees
   -g, --show-pgids    show process group ids; implies -c
  -G, --vt100         use VT100 line drawing characters
   -h, --highlight-all highlight current process and its ancestors
  -H PID, --highlight-pid=PID
                      highlight this process and its ancestors
  -l, --long          don't truncate long lines
   -n, --numeric-sort  sort output by PID
  -N TYPE, --ns-sort=TYPE
                      sort output by this namespace type
                              (cgroup, ipc, mnt, net, pid, time, user, uts)
  -p, --show-pids     show PIDs; implies -c
   -s, --show-parents  show parents of the selected process
  -S, --ns-changes    show namespace transitions
  -t, --thread-names  show full thread names
  -T, --hide-threads  hide threads, show only processes
   -u, --uid-changes   show uid transitions
  -U, --unicode       use UTF-8 (Unicode) line drawing characters
  -V, --version       display version information
   udp/tcp names: [local_port][,[rmt_host][,[rmt_port]]]

 %*s USER        PID ACCESS COMMAND
 %s is empty (not mounted ?)
 %s: Invalid option %s
 %s: no process found
 %s: unknown signal; %s -l lists signals.
 (unknown) /proc is not mounted, cannot stat /proc/self/stat.
 Bad regular expression: %s
 CPU Times
  This Process    (user system guest blkio): %6.2f %6.2f %6.2f %6.2f
  Child processes (user system guest):       %6.2f %6.2f %6.2f
 Can't get terminal capabilities
 Cannot allocate memory for matched proc: %s
 Cannot find socket's device number.
 Cannot find user %s
 Cannot open /proc directory: %s
 Cannot open /proc/net/unix: %s
 Cannot open a network socket.
 Cannot open protocol file "%s": %s
 Cannot resolve local port %s: %s
 Cannot stat %s: %s
 Cannot stat file %s: %s
 Copyright (C) 1993-2021 Werner Almesberger and Craig Small

 Copyright (C) 1993-2022 Werner Almesberger and Craig Small

 Copyright (C) 2007 Trent Waddington

 Copyright (C) 2009-2022 Craig Small

 Could not kill process %d: %s
 Error attaching to pid %i
 Invalid namespace PID Invalid namespace name Invalid option Invalid time format Kill %s(%s%d) ? (y/N)  Kill process %d ? (y/N)  Killed %s(%s%d) with signal %d
 Memory
  Vsize:       %-10s
  RSS:         %-10s 		 RSS Limit: %s
  Code Start:  %#-10lx		 Code Stop:  %#-10lx
  Stack Start: %#-10lx
  Stack Pointer (ESP): %#10lx	 Inst Pointer (EIP): %#10lx
 Namespace option requires an argument. No process specification given No processes found.
 No such user name: %s
 PSmisc comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under
the terms of the GNU General Public License.
For more information about these matters, see the files named COPYING.
 Page Faults
  This Process    (minor major): %8lu  %8lu
  Child Processes (minor major): %8lu  %8lu
 Press return to close
 Process %d not found.
 Process with pid %d does not exist.
 Process, Group and Session IDs
  Process ID: %d		  Parent ID: %d
    Group ID: %d		 Session ID: %d
  T Group ID: %d

 Process: %-14s		State: %c (%s)
  CPU#:  %-3d		TTY: %s	Threads: %ld
 Scheduling
  Policy: %s
  Nice:   %ld 		 RT Priority: %ld %s
 Signal %s(%s%d) ? (y/N)  Specified filename %s does not exist.
 Specified filename %s is not a mountpoint.
 TERM is not set
 Unable to allocate memory for proc_info
 Unable to open stat file for pid %d (%s)
 Unable to scan stat file Unknown local port AF %d
 Usage: fuser [-fIMuvw] [-a|-s] [-4|-6] [-c|-m|-n SPACE]
             [-k [-i] [-SIGNAL]] NAME...
       fuser -l
       fuser -V
Show which processes use the named files, sockets, or filesystems.

  -a,--all              display unused files too
  -i,--interactive      ask before killing (ignored without -k)
  -I,--inode            use always inodes to compare files
  -k,--kill             kill processes accessing the named file
  -l,--list-signals     list available signal names
  -m,--mount            show all processes using the named filesystems or
                        block device
  -M,--ismountpoint     fulfill request only if NAME is a mount point
  -n,--namespace SPACE  search in this name space (file, udp, or tcp)
  -s,--silent           silent operation
  -SIGNAL               send this signal instead of SIGKILL
  -u,--user             display user IDs
  -v,--verbose          verbose output
  -w,--writeonly        kill only processes with write access
  -V,--version          display version information
 Usage: killall [OPTION]... [--] NAME...
 Usage: peekfd [-8] [-n] [-c] [-d] [-V] [-h] <pid> [<fd> ..]
    -8, --eight-bit-clean        output 8 bit clean streams.
    -n, --no-headers             don't display read/write from fd headers.
    -c, --follow                 peek at any new child processes too.
    -t, --tgid                   peek at all threads where tgid equals <pid>.
    -d, --duplicates-removed     remove duplicate read/writes from the output.
    -V, --version                prints version info.
    -h, --help                   prints this help.

  Press CTRL-C to end output.
 Usage: prtstat [options] PID ...
       prtstat -V
Print information about a process
    -r,--raw       Raw display of information
    -V,--version   Display version information and exit
 Usage: pstree [-acglpsStTuZ] [ -h | -H PID ] [ -n | -N type ]
              [ -A | -G | -U ] [ PID | USER ]
   or: pstree -V
 You can only use files with mountpoint options You cannot search for only IPv4 and only IPv6 sockets at the same time You must provide at least one PID. all option cannot be used with silent option. asprintf in print_stat failed.
 disk sleep fuser (PSmisc) %s
 killall: %s lacks process entries (not mounted ?)
 killall: Bad regular expression: %s
 killall: Cannot get UID from process status
 killall: Maximum number of names is %d
 killall: skipping partial match %s(%d)
 paging peekfd (PSmisc) %s
 procfs file for %s namespace not available
 prtstat (PSmisc) %s
 pstree (PSmisc) %s
 running sleeping traced unknown zombie Project-Id-Version: psmisc 23.6-rc1
Report-Msgid-Bugs-To: csmall@dropbear.xyz
PO-Revision-Date: 2022-12-08 21:51+0700
Last-Translator: Andika Triwidada <andika@gmail.com>
Language-Team: Indonesian <translation-team-id@lists.sourceforge.net>
Language: id
MIME-Version: 1.0
Content-Type: text/plain; charset=ISO-8859-1
Content-Transfer-Encoding: 8bit
Plural-Forms: nplurals=1; plural=0;
X-Bugs: Report translation errors to the Language-Team address.
X-Generator: Poedit 3.2.2
 
  PID      mulai dari PID ini; baku adalah 1 (init)
  PENGGUNA tampilkan hanya proses yang berakar dari pengguna ini

 
Menampilkan pohon proses.

        killall -l, --list
       killall -V, --version

  -e,--exact          membutuhkan pencocokan tepat untuk setiap nama panjang
  -I,--ignore-case    pencocokan nama proses tidak memperhatikan besar huruf
  -g,--process-group  hentikan proses grup daripada proses
  -y,--younger-than   hentikan proses lebih muda dari WAKTU
  -o,--older-than     hentikan proses lebih tua dari WAKTU
  -i,--interactive    tanya untuk konfirmasi sebelum menghentikan
  -l,--list           daftar seluruh nama sinyal yang diketahui
  -q,--quiet          jangan tampilkan komplain
  -r,--regexp         interpretasikan NAMA sebagai sebuah ekstensi ekpresi regular
  -s,--signal SINYAL  kirim sinyal ini daripada SIGTERM
  -u,--user PENGGUNA  hentikan hanya proses yang berjalan sebagai PENGGUNA
  -v,--verbose        laporkan jika sinyal telah secara sukses dikirimkan
  -V,--version        tampilkan informasi versi
  -w,--wait           tunggu untuk proses untuk mati
  -n,--ns PID         cocokkan dengan proses milik ruang nama yang sama
                      dengan PID
   -4,--ipv4             cari di socket IPv4 saja
  -6,--ipv6             cari di socket IPv6 saja
   -C, --color=TIPE    warnai proses berdasarkan
                      atribut (umur)
   -Z, --security-context
                      tampilkan atribut keamanan
   -Z,--context REGEXP hanya hentikan proses yang memiliki konteks
                      (harus mendahului argumen lain)
   -a, --arguments     menampilkan argumen baris perintah
  -A, --ascii         gunakan karakter menggambar garis ASCII
  -c, --compact-not   jangan satukan sub pohon identik
   -g, --show-pgids    tampilkan id grup proses; menyiratkan -c
  -G, --vt100         gunakan karakter menggambar garis VT100
   -h, --highlight-all sorot proses saat ini dan moyangnya
  -H PID, --highlight-pid=PID
                      sorot proses ini dan moyangnya
  -l, --long          jangan potong baris panjang
   -n, --numeric-sort  urut keluaran berdasarkan PID
  -N TYPE, --ns-sort=TYPE
                      urut keluaran berdasarkan tipe ruang nama ini
                              (cgroup, ipc, mnt, net, pid, time, user, uts)
  -p, --show-pids     tampilkan PID; menyiratkan -c
   -s, --show-parents  tampilkan induk dari proses yang dipilih
  -S, --ns-changes    tampilkan transisi ruang nama
  -t, --thread-names  tampilkan nama-nama thread lengkap
  -T, --hide-threads  sembunyikan thread, hanya tampilkan proses
   -u, --uid-changes   tampilkan transisi uid
  -U, --unicode       gunakan karakter menggambar garis UTF-8 (Unicode)
  -V, --version       tampilkan informasi versi
   nama udp/tcp: [port_lokal][,[host_jauh][,[port_jauh]]]

 %*s PENGGUNA    PID AKSES  PERINTAH
 %s kosong (belum di-mount ?)
 %s: Opsi tidak valid %s
 %s: tidak ada proses yang ditemukan
 %s: sinyal tidak diketahui; %s -l daftar sinyal.
 (tak diketahui) /proc belum dikait, tidak bisa men-stat /proc/self/stat.
 Ekspresi reguler buruk: %s
 Waktu CPU
  Proses Ini   (pengguna sistem tamu blkio): %6.2f %6.2f %6.2f %6.2f
  Proses Anak  (pengguna sistem tamu):       %6.2f %6.2f %6.2f
 Tidka dapat memperoleh kapabilitas terminal
 Tidak dapat mengalokasikan memori untuk proc yang sesuai: %s
 Tidak dapat menemukan nomor perangkat socket.
 Tidak dapat menemukan pengguna %s
 Tidak dapat membuka direktori /proc: %s
 Tidak dapat membuka /proc/net/unix: %s
 Tidak dapat membuka sebuah socket jaringan.
 Tidak dapat membuka berkas protokol "%s": %s
 Tidak dapat mengurai port lokal %s: %s
 Tidak dapat men-stat %s: %s
 Tidak dapat men-stat berkas %s: %s
 Hak Cipta (C) 1993-2021 Werner Almesberger dan Craig Small

 Hak Cipta (C) 1993-2022 Werner Almesberger dan Craid Small

 Hak Cipta (C) 2007 Trent Waddington

 Hak Cipta (C) 2009-2022 Craig Small

 Tidak dapat menghentikan proses %d: %s
 Galat saat mencantol ke pid %i
 PID nama ruang tidak valid Nama ruang tidak valid Opsi tidak valid Format waktu tidak valid Bunuh %s(%s%d) ? (y/N)  Bunuh proses %d ? (y/N)  Terhenti %s(%s%d) dengan sinyal %d
 Memori
  Vsize:       %-10s
  RSS:         %-10s 		 Batas RSS: %s
  Awal Kode:   %#-10lx		 Akhir Kode: %#-10lx
  Awal Stack:  %#-10lx
  Penunjuk Stack (ESP): %#10lx	 Penunjuk Inst. (EIP): %#10lx
 Opsi nama ruang membutuhkan sebuah argumen. Tidak ada spesifikasi proses yang diberikan Tidak ada proses yang ditemukan.
 Tidak ada nama pengguna seperti itu: %s
 PSmisc datang dengan SECARA ABSOLUT TIDAK ADA GARANSI.
Ini adalah aplikasi bebas, Anda diperbolehkan untuk meredistribusikannya
di bawah ketentuan dari GNU General Public License.
Untuk informasi mengenai masalah ini, lihat berkas bernama COPYING.
 Kesalahan Page
  Proses Ini      (minor major): %8lu  %8lu
  Proses Anak     (minor major): %8lu  %8lu
 Tekan Enter untuk menutup
 Proses %d tidak ditemukan.
 Proses dengan pid %d tidak ada.
 ID Proses, Grup, dan Sesi
  ID Proses: %d		 ID Induk: %d
    ID Grup: %d		  ID Sesi: %d
  ID Grup T: %d

 Proses : %-14s		Keadaan: %c (%s)
  CPU# : %-3d		TTY: %s	Thread: %ld
 Penjadwalan
  Kebijakan: %s
  Nice:      %ld 		 Prioritas RT: %ld %s
 Sinyal %s(%s%d) ? (y/N)  Nama berkas %s yang dispesifikasikan tidak ada.
 Nama berkas %s yang dispesifikasikan bukan titik kait.
 TERM tidak diset
 Tidak dapat mengalokasikan memori untuk proc_info
 Tidak dapat membuka stat berkas untuk pid %d (%s)
 Tidak dapat memindai stat berkas Port lokal AF %d tidak diketahui
 Penggunaan: fuser [-fIMuvw] [-a|-s] [-4|-6] [-c|-m|-n RUANG]
                  [-k [-i] [-SIGNAL]] NAMA...
            fuser -l
            fuser -V
Tampilkan proses yang menggunakan nama berkas, socket, atau sistem berkas.

  -a,--all              tampilkan berkas yang tidak digunakan juga
  -i,--interactive      tanya sebelum menghentikan (abaikan tanpa -k)
  -I,--inode            selalu gunakan inode untuk membandingkan berkas
  -k,--kill             hentikan proses yang mengakses berkas bernama
  -l,--list-signals     daftar nama sinyal yang tersedia
  -m,--mount            tampilkan seluruh proses menggunakan sistem berkas bernama
                        atau peranti blok
  -M,--ismountpoint     penuhi permintaan hanya jika NAMA adalah sebuah titik kait
  -n,--namespace RUANG  cari di ruang nama ini (berkas, udp, atau tcp)
  -s,--silent           beroperasi secara sunyi
  -SIGNAL               kirim sinyal ini daripada SIGKILL
  -u,--user             tampilkan ID pengguna
  -v,--verbose          keluaran ramai
  -w,--writeonly        hanya matikan proses dengan akses tulis
  -V,--version          tampilkan informasi versi
 Penggunaan: killall [OPSI]... [--] NAMA...
 Penggunaan: peekfd [-8] [-n] [-c] [-d] [-V] [-h] <pid> [<fd> ..]
    -8, --eight-bit-clean        keluarkan 8 bit stream bersih.
    -n, --no-headers             jangan tampilkan baca/tulis dari header fd.
    -c, --follow                 lihat di proses anak baru apa pun juga.
    -t, --tgid                   peek di semua thread dengan tgid sama dengan <pid>.
    -d, --duplicates-removed     hapus duplikasi baca/tulis dari keluaran.
    -V, --version                tampilkan informasi versi.
    -h, --help                   tampilkan bantuan ini.

  Tekan CTRL-C untuk mengakhiri keluaran.
 Penggunaan: prstat [opsi] PID ...
            prstat -V
Tampilkan informasi mengenai sebuah proses
    -r,--raw        Tampilkan informasi mentah
    -V,--version    Tampilkan informasi versi dan keluar
 Cara pakai: pstree [-acglpsStTuZ] [ -h | -H PID ] [ -n | -N type ]
                   [ -A | -G | -U ] [ PID | USER ]
   atau: pstree -V
 Anda hanya dapat menggunakan berkas dengan opsi titik kait Anda tidak dapat mencari soket hanya untuk IPv4 dan hanya untuk IPv6 di waktu yang sama Anda harus menyediakan paling tidak satu PID. semua opsi tidak dapat digunakan dengan opsi silent. asprintf dalam print_stat gagal.
 disk tidur fuser (PSmisc) %s
 killall: %s tidak ada entri proses (belum di-mount ?)
 killall: Ekspresi reguler buruk: %s
 killall: Tidak dapat memperoleh UID dari status proses
 killall: Cacah maksimal dari nama adalah %d
 killall: melewatkan pencocokan sebagian %s(%d)
 paging peekfd (PSmisc) %s
 berkas procfs bagi ruang nama %s tidak tersedia
 prtstat (PSmisc) %s
 pstree (PSmisc) %s
 berjalan tertidur terlacak tidak diketahui zombie 