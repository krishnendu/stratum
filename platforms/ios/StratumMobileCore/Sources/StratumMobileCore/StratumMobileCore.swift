import Foundation

public enum StratumMobile {
  public static var version: String {
    String(cString: stratum_mobile_version())
  }
}
