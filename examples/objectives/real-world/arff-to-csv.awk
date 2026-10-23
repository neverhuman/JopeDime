# Convert a dense ARFF file to a header CSV: attribute names become the header; quotes are
# removed from nominal and string values. Values containing commas are not supported.
BEGIN { IGNORECASE = 1; data = 0; n = 0 }
/^[[:space:]]*%/ || /^[[:space:]]*$/ { next }
!data && tolower($1) == "@attribute" {
  name = $2; gsub(/^['"]|['"]$/, "", name); names[++n] = name; next
}
!data && tolower($1) == "@data" {
  header = names[1]; for (i = 2; i <= n; i++) header = header "," names[i]; print header
  data = 1; next
}
data { gsub(/'/, ""); gsub(/"/, ""); print }
